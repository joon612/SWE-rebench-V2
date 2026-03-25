#!/usr/bin/env python3
"""
Batch-generate (and optionally build) Dockerfiles from a JSONL input file.

Each JSONL line is expected to contain at least:
  - repo            : "owner/repo"
  - base_commit_sha : git commit SHA
  - setup_command   : shell commands to install / build the project
  - test_command    : shell commands to run the test suite
  - runtime         : {"os": [...], "stack": [{"lang": ..., "version": ...}, ...]}

The language (and therefore base image) is resolved from runtime.stack[0].lang.
If no matching base image exists, all available images are tried and the first
entry in the list that matches the stack is used.

The JSONL may optionally include:
  - instance_id     : explicit ID; derived from repo + commit if absent
  - image_name      : explicit base image tag; auto-resolved if absent

Usage
-----
    python scripts/batch_generate_dockerfiles.py \\
        --jsonl path/to/input.jsonl \\
        --output-dir dockerfiles/ \\
        [--base-image-registry myregistry] \\
        [--base-image-tag-suffix _base] \\
        [--workers 8] \\
        [--build] \\
        [--image-registry myregistry] \\
        [--tag-prefix instance-] \\
        [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    print("Missing dependency: jinja2. Install with `pip install jinja2`.", file=sys.stderr)
    raise

# ---------------------------------------------------------------------------
# Base-image resolution  (mirrors generate_dockerfile.py)
# ---------------------------------------------------------------------------

BASE_IMAGES: dict[str, list[str]] = {
    "python":  ["python_3.11", "python_3.9", "python_3.7"],
    "go":      ["go_1.23.8", "go_1.22.12", "go_1.21.13", "go_1.20.14", "go_1.19.13", "go_1.18.10"],
    "rust":    ["rust_1.84", "rust_1.83", "rust_1.81", "rust_1.79", "rust_1.78", "rust_1.77"],
    "node":    ["node_20", "node_18", "node_16"],
    "javascript": ["node_20", "node_18", "node_16"],
    "java":    ["java_21", "java_17", "java_11"],
    "ruby":    ["ruby_3.3", "ruby_3.1"],
    "c":       ["c"],
    "cpp":     ["c"],
    "csharp":  ["csharp"],
    "dotnet":  ["csharp"],
    "elixir":  ["elixir"],
    "dart":    ["dart"],
    "scala":   ["scala"],
    "kotlin":  ["kotlin"],
    "julia":   ["julia"],
    "lua":     ["lua"],
    "php":     ["php_8.3.16"],
    "r":       ["r"],
    "swift":   ["swift"],
    "clojure": ["clojure"],
    "ocaml":   ["ocaml"],
}

DEFAULT_TAG_SUFFIX = "_base"

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_base_image(lang: str | None, version: str | None, tag_suffix: str) -> str:
    """Return the best-matching base image tag for (lang, version)."""
    lang_key = (lang or "").lower()
    candidates = BASE_IMAGES.get(lang_key, [])
    if not candidates:
        # Unknown language: fall back to python_3.11
        candidates = ["python_3.11"]

    if version:
        # Find a candidate whose name contains the major version number
        major = version.split(".")[0]
        preferred = [c for c in candidates if f"_{major}" in c or c.endswith(f"_{major}")]
        if preferred:
            candidates = preferred

    return f"{candidates[0]}{tag_suffix}:latest"


def make_instance_id(repo: str, commit: str) -> str:
    slug = repo.replace("/", "__").lower()
    short = (commit or "unknown")[:7]
    return f"{slug}-{short}"


def parse_install_commands(script: str) -> list[str]:
    """
    Convert a multi-line bash script into a list of individual commands
    suitable for embedding in a Dockerfile RUN layer.

    Lines that are:
      - empty / whitespace-only
      - pure comments  (#...)
      - shebang  (#!/...)
    are dropped.  The leading comment that specifies language (e.g. ``#python``)
    is also dropped.  ``set -euo pipefail`` is kept.
    """
    cmds: list[str] = []
    lines = script.splitlines()
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#!"):        # shebang
            continue
        if stripped.startswith("#"):         # comment or language marker
            continue
        cmds.append(stripped)
    return cmds


def render_dockerfile(template_path: Path, spec: dict, registry: str) -> str:
    env = Environment(loader=FileSystemLoader(str(template_path.parent)), autoescape=False)
    template = env.get_template(template_path.name)
    return template.render(spec=spec, base_image_registry=registry, platform="linux/amd64")


def build_image(dockerfile: Path, context_dir: Path, tag: str) -> None:
    cmd = ["docker", "build", "-f", str(dockerfile), "-t", tag, str(context_dir)]
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Per-record processing
# ---------------------------------------------------------------------------

def process_record(
    record: dict,
    template_path: Path,
    output_dir: Path,
    base_registry: str,
    tag_suffix: str,
    do_build: bool,
    image_registry: str,
    tag_prefix: str,
    dry_run: bool,
) -> tuple[str, bool, str]:
    """
    Process one JSONL record.

    Returns (instance_id, success, error_message).
    """
    repo = record.get("repo", "")
    commit = record.get("base_commit_sha", "")
    setup_script = record.get("setup_command", "")
    test_script = record.get("test_command", "")

    # Derive instance_id
    instance_id = record.get("instance_id") or make_instance_id(repo, commit)

    # Resolve base image
    if "image_name" in record:
        image_name = record["image_name"]
    else:
        stack = record.get("runtime", {}).get("stack", [])
        lang = stack[0].get("lang") if stack else None
        version = stack[0].get("version") if stack else None
        image_name = resolve_base_image(lang, version, tag_suffix)

    # Parse install / test commands
    install_cmds = parse_install_commands(setup_script)
    test_cmds = parse_install_commands(test_script)

    install_config = {
        "image_name":  image_name,
        "install":     install_cmds,
        "test_cmd":    test_cmds,
        "log_parser":  record.get("log_parser", ""),
    }

    spec = {
        "repo":           repo,
        "base_commit":    commit,
        "instance_id":    instance_id,
        "install_config": install_config,
    }

    try:
        content = render_dockerfile(template_path, spec, base_registry)
    except Exception as exc:
        return instance_id, False, f"Template render error: {exc}"

    dockerfile_path = output_dir / f"{instance_id}.Dockerfile"

    if dry_run:
        _log(f"[DRY-RUN] Would write {dockerfile_path}")
        return instance_id, True, ""

    try:
        dockerfile_path.write_text(content.replace("\r\n", "\n"), encoding="utf-8", newline="\n")
    except OSError as exc:
        return instance_id, False, f"Write error: {exc}"

    _log(f"[OK] Rendered  {dockerfile_path}")

    if do_build:
        tag_parts = [tag_prefix, instance_id]
        tag = "".join(tag_parts)
        if image_registry:
            tag = f"{image_registry}/{tag}"
        try:
            build_image(dockerfile_path, output_dir, tag)
            _log(f"[OK] Built     {tag}")
        except subprocess.CalledProcessError as exc:
            return instance_id, False, f"docker build failed (exit {exc.returncode})"

    return instance_id, True, ""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch-generate Dockerfiles from a JSONL file (parallel).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--jsonl", required=True, help="Path to the input .jsonl file.")
    parser.add_argument(
        "--template",
        default="combine.Dockerfile.j2",
        help="Path to the Jinja2 Dockerfile template (default: combine.Dockerfile.j2).",
    )
    parser.add_argument(
        "--output-dir",
        default="dockerfiles",
        help="Directory to write rendered Dockerfiles (default: dockerfiles/).",
    )
    parser.add_argument(
        "--base-image-registry",
        default="",
        metavar="REGISTRY",
        help="Registry/repo prefix prepended to base image names in the Dockerfile.",
    )
    parser.add_argument(
        "--base-image-tag-suffix",
        default=DEFAULT_TAG_SUFFIX,
        metavar="SUFFIX",
        help=f"Suffix appended to base image stems (default: {DEFAULT_TAG_SUFFIX}).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8).",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="Also run `docker build` for each rendered Dockerfile.",
    )
    parser.add_argument(
        "--image-registry",
        default="",
        help="Registry prefix for built instance image tags (only used with --build).",
    )
    parser.add_argument(
        "--tag-prefix",
        default="",
        help="Prefix prepended to instance image tag names (only used with --build).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate all records but do not write any files.",
    )
    parser.add_argument(
        "--summary",
        default="",
        metavar="PATH",
        help="Write a JSON summary of results to this path.",
    )
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_file():
        print(f"JSONL file not found: {jsonl_path}", file=sys.stderr)
        return 1

    template_path = Path(args.template)
    if not template_path.is_file():
        print(f"Template not found: {template_path}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load all records
    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping line {lineno} (invalid JSON): {exc}", file=sys.stderr)

    if not records:
        print("No valid records found in JSONL file.", file=sys.stderr)
        return 1

    total = len(records)
    _log(f"Processing {total} record(s) with {args.workers} worker(s) ...")

    results: list[dict] = []
    succeeded = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        future_map = {
            pool.submit(
                process_record,
                rec,
                template_path,
                output_dir,
                args.base_image_registry,
                args.base_image_tag_suffix,
                args.build,
                args.image_registry,
                args.tag_prefix,
                args.dry_run,
            ): rec
            for rec in records
        }

        for future in as_completed(future_map):
            instance_id, ok, err = future.result()
            results.append({"instance_id": instance_id, "success": ok, "error": err})
            if ok:
                succeeded += 1
            else:
                failed += 1
                _log(f"[FAIL] {instance_id}: {err}")

    _log(f"\nDone. {succeeded}/{total} succeeded, {failed} failed.")

    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            json.dumps({"total": total, "succeeded": succeeded, "failed": failed, "results": results}, indent=2),
            encoding="utf-8",
        )
        _log(f"Summary written to {summary_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
