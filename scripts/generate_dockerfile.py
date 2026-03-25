#!/usr/bin/env python3
"""
Generate a Dockerfile for a given git repository, branch, and commit.

The script:
  1. Clones the repo at the specified commit (shallow where possible).
  2. Detects the primary programming language from top-level files.
  3. Calls an LLM (OpenAI-compatible) to produce install_config JSON
     (base image, install commands, test command, log parser).
  4. Renders combine.Dockerfile.j2 with the generated config.

Usage:
    python scripts/generate_dockerfile.py \
        --repo owner/repo \
        --commit abc123def456 \
        [--branch main] \
        [--output path/to/output.Dockerfile] \
        [--base-image-registry myregistry] \
        [--model gpt-4o] \
        [--api-key KEY] \
        [--api-base URL] \
        [--dry-run]
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore[assignment]

try:
    from jinja2 import Environment, FileSystemLoader
except ImportError:
    print("Missing dependency: jinja2. Install with `pip install jinja2`.", file=sys.stderr)
    raise

try:
    from openai import OpenAI
except ImportError:
    print("Missing dependency: openai. Install with `pip install openai`.", file=sys.stderr)
    raise


# ---------------------------------------------------------------------------
# Available base images derived from base_dockerfiles/ directory
# Map language key → ordered list of image names (most to least preferred)
# ---------------------------------------------------------------------------

# OS metadata for each base image stem (name without the tag suffix).
# Used to tell the LLM which package manager to use so it never guesses.
# All current base images are Debian/Ubuntu → apt-get.
BASE_IMAGE_OS: dict[str, str] = {
    # ----- Python -----
    "python_3.11": "Debian (python:3.11-slim). Use apt-get for system packages.",
    "python_3.9":  "Debian (python:3.9-slim).  Use apt-get for system packages.",
    "python_3.7":  "Debian (python:3.7-slim).  Use apt-get for system packages.",
    # ----- Go -----
    "go_1.23.8":   "Ubuntu 22.04. Use apt-get for system packages.",
    "go_1.22.12":  "Ubuntu 22.04. Use apt-get for system packages.",
    "go_1.21.13":  "Ubuntu 22.04. Use apt-get for system packages.",
    "go_1.20.14":  "Ubuntu 22.04. Use apt-get for system packages.",
    "go_1.19.13":  "Ubuntu 22.04. Use apt-get for system packages.",
    "go_1.18.10":  "Ubuntu 22.04. Use apt-get for system packages.",
    # ----- Rust -----
    "rust_1.84":   "Debian (rust:1.84). Use apt-get for system packages.",
    "rust_1.83":   "Debian (rust:1.83). Use apt-get for system packages.",
    "rust_1.81":   "Debian (rust:1.81). Use apt-get for system packages.",
    "rust_1.79":   "Debian (rust:1.79). Use apt-get for system packages.",
    "rust_1.78":   "Debian (rust:1.78). Use apt-get for system packages.",
    "rust_1.77":   "Debian (rust:1.77). Use apt-get for system packages.",
    # ----- Node -----
    "node_20":     "Ubuntu 22.04. Use apt-get for system packages.",
    "node_18":     "Ubuntu 22.04. Use apt-get for system packages.",
    "node_16":     "Ubuntu 22.04. Use apt-get for system packages.",
    # ----- Java -----
    "java_21":     "Debian (maven:3.9-eclipse-temurin-21). Use apt-get for system packages.",
    "java_17":     "Debian (maven:3.9-eclipse-temurin-17). Use apt-get for system packages.",
    "java_11":     "Debian (maven:3.9-eclipse-temurin-11). Use apt-get for system packages.",
    # ----- Ruby -----
    "ruby_3.3":    "Debian (ruby:3.3). Use apt-get for system packages.",
    "ruby_3.1":    "Debian (ruby:3.1). Use apt-get for system packages.",
    # ----- C -----
    "c":           "Ubuntu 22.04. Use apt-get for system packages.",
    # ----- Others (all Debian/Ubuntu family) -----
    "csharp":      "Debian (dotnet/sdk:8.0). Use apt-get for system packages.",
    "elixir":      "Debian (elixir:1.16). Use apt-get for system packages.",
    "dart":        "Debian (dart:stable-sdk). Use apt-get for system packages.",
    "scala":       "Debian bookworm slim. Use apt-get for system packages.",
    "kotlin":      "Debian (eclipse-temurin:11-jdk). Use apt-get for system packages.",
    "julia":       "Debian bookworm (julia:1.10-bookworm). Use apt-get for system packages.",
    "lua":         "Debian bookworm (nickblah/lua:5.4-bookworm). Use apt-get for system packages.",
    "php_8.3.16":  "Debian (php:8.3.16). Use apt-get for system packages.",
    "r":           "Debian/Ubuntu (rocker/r-ver:4.4.1). Use apt-get for system packages.",
    "swift":       "Ubuntu 22.04 jammy (swift:5.10-jammy). Use apt-get for system packages.",
    "clojure":     "Debian (eclipse-temurin:21-jdk). Use apt-get for system packages.",
    "ocaml":       "Debian bookworm. Use apt-get for system packages.",
}

BASE_IMAGES: dict[str, list[str]] = {
    "python":  ["python_3.11", "python_3.9", "python_3.7"],
    "go":      ["go_1.23.8", "go_1.22.12", "go_1.21.13", "go_1.20.14", "go_1.19.13", "go_1.18.10"],
    "rust":    ["rust_1.84", "rust_1.83", "rust_1.81", "rust_1.79", "rust_1.78", "rust_1.77"],
    "node":    ["node_20", "node_18", "node_16"],
    "java":    ["java_21", "java_17", "java_11"],
    "ruby":    ["ruby_3.3", "ruby_3.1"],
    "c":       ["c"],
    "csharp":  ["csharp"],
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

DEFAULT_BASE_IMAGE_TAG_SUFFIX = "_base"

# ---------------------------------------------------------------------------
# Language detection heuristics
# Each entry: (list of indicator filename patterns, language key)
# Patterns starting with "*." match by file extension; others match exact names.
# Ordered from most-specific to least-specific.
# ---------------------------------------------------------------------------
LANG_INDICATORS: list[tuple[list[str], str]] = [
    (["mix.exs"],                                          "elixir"),
    (["pubspec.yaml", "pubspec.yml"],                      "dart"),
    (["dune-project", "*.opam"],                           "ocaml"),
    (["Package.swift", "*.swift"],                         "swift"),
    (["*.clj", "project.clj", "deps.edn"],                 "clojure"),
    (["Project.toml", "*.jl"],                             "julia"),
    (["go.mod"],                                           "go"),
    (["Cargo.toml"],                                       "rust"),
    (["build.sbt"],                                        "scala"),
    (["*.kt", "*.kts"],                                    "kotlin"),
    (["pom.xml", "build.gradle", "build.gradle.kts"],      "java"),
    (["setup.py", "setup.cfg", "pyproject.toml"],          "python"),
    (["package.json"],                                     "node"),
    (["Gemfile", "*.gemspec"],                             "ruby"),
    (["composer.json"],                                    "php"),
    (["*.cs", "*.csproj", "*.sln"],                        "csharp"),
    (["DESCRIPTION", "NAMESPACE"],                         "r"),
    (["CMakeLists.txt", "configure.ac", "*.c", "*.cpp"],   "c"),
    (["*.lua"],                                            "lua"),
]

# Key files to read for the LLM prompt (tried in this order)
KEY_FILE_CANDIDATES = [
    "README.md", "README.rst", "README.txt", "README",
    "CONTRIBUTING.md", "INSTALL.md",
    "Makefile", "CMakeLists.txt", "configure.ac",
    "setup.py", "setup.cfg", "pyproject.toml", "requirements.txt",
    "go.mod", "Cargo.toml",
    "package.json",
    "pom.xml", "build.gradle", "build.gradle.kts",
    "Gemfile", "mix.exs", "pubspec.yaml",
    "build.sbt", "project.clj", "deps.edn",
    ".travis.yml",
    ".github/workflows/ci.yml",
    ".github/workflows/test.yml",
    ".github/workflows/build.yml",
    "Dockerfile", "docker-compose.yml",
]

# Max total characters of key-file content sent to the LLM
_MAX_KEY_FILE_CHARS = 10_000
# Max characters read from each individual key file
_MAX_PER_FILE_CHARS = 2_000


def detect_language(repo_dir: Path) -> str | None:
    """Heuristically detect the primary language from top-level repository files."""
    top_files = {f.name for f in repo_dir.iterdir() if f.is_file()}
    top_exts = {f.suffix.lower() for f in repo_dir.iterdir() if f.is_file()}

    for indicators, lang in LANG_INDICATORS:
        for pattern in indicators:
            if pattern.startswith("*."):
                if pattern[1:] in top_exts:  # e.g. ".kt"
                    return lang
            else:
                if pattern in top_files:
                    return lang
    return None


def read_key_files(repo_dir: Path) -> str:
    """Collect content from key configuration/documentation files for the LLM prompt."""
    collected: list[str] = []
    total = 0
    for name in KEY_FILE_CANDIDATES:
        path = repo_dir / name
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        snippet = text[:_MAX_PER_FILE_CHARS]
        collected.append(f"=== {name} ===\n{snippet}")
        total += len(snippet)
        if total >= _MAX_KEY_FILE_CHARS:
            break
    return "\n\n".join(collected)


def _image_os_note(image_name: str, tag_suffix: str) -> str:
    """Return a human-readable OS/package-manager note for a base image tag like python_3.7_base:latest."""
    ref = image_name.split(":", 1)[0]  # strip :latest
    stem = ref[:-len(tag_suffix)] if tag_suffix and ref.endswith(tag_suffix) else ref
    return BASE_IMAGE_OS.get(stem, "Debian/Ubuntu family. Use apt-get for system packages.")


def build_prompt(
    repo: str,
    commit: str,
    lang: str | None,
    key_files: str,
    available_images: list[str],
    tag_suffix: str = DEFAULT_BASE_IMAGE_TAG_SUFFIX,
) -> str:
    lines = []
    for img in available_images:
        note = _image_os_note(img, tag_suffix)
        lines.append(f"  - {img}  # {note}")
    images_list = "\n".join(lines)
    lang_str = lang or "unknown (please infer from the repository files)"
    return f"""\
You are an expert DevOps engineer. Given a GitHub repository and its key files, \
generate a minimal install_config JSON for building and testing the project in Docker.

Repository: {repo}
Commit: {commit}
Detected language: {lang_str}

Available base images (choose the most appropriate one).
Each line shows the image name followed by its OS and package manager.
You MUST use that package manager in the install commands — never use a different one.
{images_list}

Key repository files:
{key_files}

Generate a JSON object with exactly these fields:

"image_name"  — chosen base image EXACTLY from the available base image list above.
                 Do not invent image names. Do not remove suffixes. Do not change tags.
"install"     — ordered list of shell commands to install/build the project.
                 Rules:
                 - Run from the project root directory.
                 - Non-interactive: DEBIAN_FRONTEND=noninteractive apt-get install -y ... for apt-get images.
                 - Do NOT include language/runtime installation (the base image already has it).
                 - Use the package manager shown next to the chosen image. NEVER use apk, yum, brew, etc.
                 - Include dependency installation, compilation/build steps, but NOT test execution.
"log_parser"  — name of the parser for test output. Choose from:
                 parse_log_pytest, parse_log_pytest_v2, parse_log_django,
                 parse_log_gotest, parse_log_cargo,
                 parse_log_jest, parse_log_jest_json, parse_log_vitest,
                 parse_log_maven, parse_log_gradle_custom,
                 parse_log_ruby_v1, parse_log_ruby_v2,
                 parse_test_report (JUnit XML),
                 parse_log_cpp, parse_log_googletest, parse_log_doctest,
                 parse_log_phpunit, parse_log_scala, parse_log_dart,
                 parse_log_elixir, parse_log_ocaml, parse_log_swift,
                 parse_log_csharp, parse_log_tap, parse_log_jq,
                 parse_log_r, parse_log_julia, parse_log_lein, parse_log_sbt
"test_cmd"    — list of shell command(s) to run the full test suite.
                 Must produce verbose per-test output (show individual test names).
                 Disable ANSI color when possible.

Return ONLY a valid JSON object with these four fields — no other text.
"""


def get_available_base_images(lang: str | None, tag_suffix: str) -> list[str]:
    if lang:
        candidates = BASE_IMAGES.get(lang, [])
    else:
        candidates = [img for imgs in BASE_IMAGES.values() for img in imgs]
    return [f"{name}{tag_suffix}:latest" for name in candidates]


def normalize_image_name(image_name: str, available_images: list[str], tag_suffix: str) -> str:
    normalized = image_name.strip()
    if normalized in available_images:
        return normalized

    if ":" not in normalized and f"{normalized}:latest" in available_images:
        return f"{normalized}:latest"

    # Accept an unsuffixed model answer like python_3.11:latest and map it to python_3.11_base:latest.
    base_name, _, tag = normalized.partition(":")
    tag = tag or "latest"
    suffixed = f"{base_name}{tag_suffix}:{tag}"
    if suffixed in available_images:
        return suffixed

    raise ValueError(
        f"Unsupported image_name '{image_name}'. Expected one of: {', '.join(available_images)}"
    )


def docker_image_exists(tag: str) -> bool:
    result = subprocess.run(
        ["docker", "image", "inspect", tag],
        capture_output=True,
    )
    return result.returncode == 0


def derive_base_dockerfile_path(image_name: str, tag_suffix: str) -> Path:
    image_ref = image_name.split(":", 1)[0]
    stem = image_ref[:-len(tag_suffix)] if tag_suffix and image_ref.endswith(tag_suffix) else image_ref
    return Path("base_dockerfiles") / f"Dockerfile_{stem}"


def ensure_base_image(image_name: str, tag_suffix: str, platform: str = "linux/amd64") -> None:
    if docker_image_exists(image_name):
        return

    dockerfile_path = derive_base_dockerfile_path(image_name, tag_suffix)
    if not dockerfile_path.is_file():
        raise FileNotFoundError(
            f"Base image '{image_name}' is missing locally, and its Dockerfile was not found at {dockerfile_path}."
        )

    print(
        f"Base image {image_name} not found locally. Building it from {dockerfile_path} ...",
        file=sys.stderr,
    )
    cmd = [
        "docker", "build", "-f", str(dockerfile_path), "-t", image_name,
        "--platform", platform, str(dockerfile_path.parent),
    ]
    subprocess.run(cmd, check=True)


def call_llm(client: OpenAI, model: str, prompt: str) -> str:
    """Call the LLM and return the raw text response."""
    # Attempt structured JSON output (supported by modern OpenAI models)
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert DevOps engineer. "
                        "Always respond with a single valid JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
    except Exception:
        # Fallback: call without response_format for models that don't support it
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert DevOps engineer. "
                        "Always respond with a single valid JSON object."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
    return (completion.choices[0].message.content or "").strip()


def extract_json(text: str) -> dict:
    """Extract and parse a JSON object from LLM response text."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Try to find a ```json ... ``` block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        return json.loads(m.group(1))
    # Try to find the outermost { ... }
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        return json.loads(m.group(0))
    raise ValueError(f"Could not extract JSON from LLM response:\n{text[:500]}")


def _detect_proxy() -> str | None:
    """Return an HTTPS proxy URL from common environment variables, or None."""
    for var in ("https_proxy", "HTTPS_PROXY", "http_proxy", "HTTP_PROXY", "ALL_PROXY"):
        val = os.environ.get(var, "").strip()
        if val:
            return val
    return None


def _run_git(cmd: list[str], proxy: str | None = None) -> None:
    """Run a git command, printing stderr and raising on failure."""
    if proxy:
        # Inject proxy as a git -c option right after 'git'
        cmd = [cmd[0], "-c", f"https.proxy={proxy}", "-c", f"http.proxy={proxy}"] + cmd[1:]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        msg = result.stderr.decode(errors="replace").strip()
        if msg:
            print(f"git error: {msg}", file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd, stderr=msg)


def clone_repo(repo: str, commit: str, branch: str | None, dest: Path, proxy: str | None = None) -> None:
    """
    Clone the GitHub repository and checkout the specified commit.

    Uses a shallow clone for speed; falls back to a deeper fetch if the
    commit is not present after the initial clone.
    If a branch is specified but doesn't exist, retries without --branch.
    """
    url = f"https://github.com/{repo}"

    def _clone(with_branch: bool) -> None:
        cmd = ["git", "clone", "--quiet", "--depth=50"]
        if with_branch and branch:
            cmd += ["--branch", branch]
        cmd += [url, str(dest)]
        _run_git(cmd, proxy=proxy)

    if branch:
        try:
            _clone(with_branch=True)
        except subprocess.CalledProcessError as exc:
            print(
                f"Warning: clone with --branch {branch} failed "
                f"({exc.stderr}). Retrying without --branch ...",
                file=sys.stderr,
            )
            # Clean up partial clone directory before retrying
            if dest.exists():
                shutil.rmtree(dest)
            _clone(with_branch=False)
    else:
        _clone(with_branch=False)

    # Try to reset to the requested commit
    result = subprocess.run(
        ["git", "-C", str(dest), "reset", "--hard", commit],
        capture_output=True,
    )
    if result.returncode != 0:
        # Commit not reachable in shallow history; deepen the clone
        _run_git(["git", "-C", str(dest), "fetch", "--unshallow"], proxy=proxy)
        _run_git(["git", "-C", str(dest), "reset", "--hard", commit], proxy=proxy)


# YAML keys that map directly to argparse dest names
_YAML_KEYS = {
    "repo", "commit", "branch", "output", "template",
    "base_image_registry", "model", "api_key", "api_base",
    "dry_run", "clone_dir", "no_clone", "base_image_tag_suffix",
    "ensure_base_image", "git_proxy",
}


def load_config(path: Path) -> dict:
    """Load a YAML config file and return a dict of option values.

    YAML keys use underscores (matching argparse dest names).
    Keys not in _YAML_KEYS are silently ignored.
    """
    if yaml is None:
        print(
            "Error: PyYAML is not installed. Run: pip install pyyaml",
            file=sys.stderr,
        )
        sys.exit(1)
    text = path.read_text(encoding="utf-8")
    data = yaml.safe_load(text) or {}
    if not isinstance(data, dict):
        print(f"Error: config file must be a YAML mapping: {path}", file=sys.stderr)
        sys.exit(1)
    return {k: v for k, v in data.items() if k in _YAML_KEYS}


def render_dockerfile(template_path: Path, spec: dict, registry: str) -> str:
    env = Environment(loader=FileSystemLoader(str(template_path.parent)), autoescape=False)
    template = env.get_template(template_path.name)
    return template.render(spec=spec, base_image_registry=registry, platform="linux/amd64")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a Dockerfile for a given git repo, branch, and commit.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        metavar="FILE",
        help="Path to a YAML config file. CLI flags override config file values.",
    )
    parser.add_argument(
        "--repo",
        default=None,
        metavar="OWNER/REPO",
        help="GitHub repository in owner/repo format (e.g. torvalds/linux).",
    )
    parser.add_argument(
        "--commit",
        default=None,
        metavar="SHA",
        help="Full or abbreviated git commit SHA.",
    )
    parser.add_argument(
        "--branch",
        default=None,
        metavar="BRANCH",
        help="Branch name (optional; speeds up shallow clone).",
    )
    parser.add_argument(
        "--output",
        default=None,
        metavar="PATH",
        help="Output Dockerfile path (default: <owner>__<repo>-<commit[:7]>.Dockerfile).",
    )
    parser.add_argument(
        "--template",
        default="combine.Dockerfile.j2",
        help="Path to the Jinja2 Dockerfile template (default: combine.Dockerfile.j2).",
    )
    parser.add_argument(
        "--base-image-registry",
        default="",
        metavar="REGISTRY",
        help="Registry/repo prefix for base images (e.g. docker.io/myorg).",
    )
    parser.add_argument(
        "--base-image-tag-suffix",
        default=DEFAULT_BASE_IMAGE_TAG_SUFFIX,
        metavar="SUFFIX",
        help=(
            "Suffix used by locally built base images from base_dockerfiles "
            f"(default: {DEFAULT_BASE_IMAGE_TAG_SUFFIX})."
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="OpenAI-compatible model name (default: gpt-4o).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="OpenAI API key (default: OPENAI_API_KEY environment variable).",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        metavar="URL",
        help="OpenAI-compatible API base URL (for custom endpoints).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the prompt that would be sent to the LLM and exit without calling it.",
    )
    parser.add_argument(
        "--clone-dir",
        default=None,
        metavar="DIR",
        help=(
            "Directory to clone into. "
            "If not set, a temporary directory is used and cleaned up afterward."
        ),
    )
    parser.add_argument(
        "--no-clone",
        action="store_true",
        help=(
            "Skip cloning; use --clone-dir as the already-checked-out repo root. "
            "Useful when the repo is already available locally."
        ),
    )
    parser.add_argument(
        "--ensure-base-image",
        action="store_true",
        help="Automatically build the required base image from base_dockerfiles if it is missing locally.",
    )
    parser.add_argument(
        "--git-proxy",
        default=None,
        metavar="URL",
        help=(
            "HTTP/HTTPS proxy for git clone (e.g. http://127.0.0.1:7890). "
            "If not set, auto-detected from https_proxy / HTTPS_PROXY / http_proxy env vars."
        ),
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------
    # Merge config file (lowest priority) → CLI flags (highest priority)
    # -----------------------------------------------------------------
    if args.config:
        config_path = Path(args.config)
        if not config_path.is_file():
            print(f"Config file not found: {config_path}", file=sys.stderr)
            return 1
        cfg = load_config(config_path)
        for key, value in cfg.items():
            # Only apply config value when the CLI flag was not explicitly set.
            current = getattr(args, key, None)
            if current == parser.get_default(key):
                setattr(args, key, value)

    # Validate required fields (may now come from config file)
    missing = [f"--{f}" for f in ("repo", "commit") if not getattr(args, f, None)]
    if missing:
        parser.error(
            f"the following arguments are required (provide via CLI or config file): "
            f"{', '.join(missing)}"
        )

    # Validate template
    template_path = Path(args.template)
    if not template_path.is_file():
        print(f"Template not found: {template_path}", file=sys.stderr)
        return 1

    # Derive instance_id and output path
    repo_slug = args.repo.replace("/", "__").lower()
    short_commit = args.commit[:7]
    instance_id = f"{repo_slug}-{short_commit}"
    output_path = Path(args.output) if args.output else Path(f"{instance_id}.Dockerfile")

    # Set up clone directory (temp or user-specified)
    tmpdir_obj = None
    if args.no_clone:
        if not args.clone_dir:
            print("--no-clone requires --clone-dir to point at the repo root.", file=sys.stderr)
            return 1
        clone_dir = Path(args.clone_dir)
    elif args.clone_dir:
        clone_dir = Path(args.clone_dir)
    else:
        tmpdir_obj = tempfile.TemporaryDirectory()
        clone_dir = Path(tmpdir_obj.name) / "repo"

    try:
        # -----------------------------------------------------------------
        # Step 1: Clone
        # -----------------------------------------------------------------
        if not args.no_clone:
            print(f"Cloning https://github.com/{args.repo} @ {args.commit} ...", file=sys.stderr)
            clone_dir.mkdir(parents=True, exist_ok=True)
            proxy = args.git_proxy or _detect_proxy()
            if proxy:
                print(f"Using git proxy: {proxy}", file=sys.stderr)
            clone_repo(args.repo, args.commit, args.branch, clone_dir, proxy=proxy)
            print("Clone complete.", file=sys.stderr)
        else:
            if not clone_dir.is_dir():
                print(f"Clone directory not found: {clone_dir}", file=sys.stderr)
                return 1

        # -----------------------------------------------------------------
        # Step 2: Detect language
        # -----------------------------------------------------------------
        lang = detect_language(clone_dir)
        print(f"Detected language: {lang or 'unknown'}", file=sys.stderr)

        available_images = get_available_base_images(lang, args.base_image_tag_suffix)
        if not available_images:
            # Fall back: let the LLM pick from all images
            available_images = get_available_base_images(None, args.base_image_tag_suffix)
            if lang:
                print(
                    f"Warning: no base images registered for language '{lang}'. "
                    "Passing all available images to the LLM.",
                    file=sys.stderr,
                )

        # -----------------------------------------------------------------
        # Step 3: Read key files
        # -----------------------------------------------------------------
        key_files = read_key_files(clone_dir)
        if not key_files:
            print(
                "Warning: no key files found in repository root. "
                "LLM will rely on language detection only.",
                file=sys.stderr,
            )

        # -----------------------------------------------------------------
        # Step 4: Build prompt
        # -----------------------------------------------------------------
        prompt = build_prompt(args.repo, args.commit, lang, key_files, available_images, args.base_image_tag_suffix)

        if args.dry_run:
            print("=== LLM PROMPT (dry-run, not sent) ===\n")
            print(prompt)
            return 0

        # -----------------------------------------------------------------
        # Step 5: Call LLM
        # -----------------------------------------------------------------
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(
                "Error: no API key provided. Use --api-key or set OPENAI_API_KEY.",
                file=sys.stderr,
            )
            return 1

        client_kwargs: dict = {"api_key": api_key}
        if args.api_base:
            client_kwargs["base_url"] = args.api_base

        client = OpenAI(**client_kwargs)
        print(f"Calling {args.model} to generate install_config ...", file=sys.stderr)
        raw_response = call_llm(client, args.model, prompt)

        # -----------------------------------------------------------------
        # Step 6: Parse LLM response
        # -----------------------------------------------------------------
        try:
            install_config = extract_json(raw_response)
        except (ValueError, json.JSONDecodeError) as exc:
            print(f"Error: failed to parse LLM response as JSON: {exc}", file=sys.stderr)
            print("Raw LLM response:\n", raw_response, file=sys.stderr)
            return 1

        # Validate required fields
        for field in ("image_name", "install", "test_cmd"):
            if field not in install_config:
                print(
                    f"Error: LLM response is missing required field '{field}'.",
                    file=sys.stderr,
                )
                print("Parsed install_config:\n", json.dumps(install_config, indent=2), file=sys.stderr)
                return 1

        try:
            install_config["image_name"] = normalize_image_name(
                install_config["image_name"],
                available_images,
                args.base_image_tag_suffix,
            )
        except (KeyError, TypeError, ValueError) as exc:
            print(f"Error: invalid image_name in LLM response: {exc}", file=sys.stderr)
            print("Parsed install_config:\n", json.dumps(install_config, indent=2), file=sys.stderr)
            return 1

        if args.ensure_base_image:
            try:
                ensure_base_image(install_config["image_name"], args.base_image_tag_suffix)
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                print(f"Error while ensuring base image: {exc}", file=sys.stderr)
                return 1

        print("Generated install_config:", file=sys.stderr)
        print(json.dumps(install_config, indent=2), file=sys.stderr)

    finally:
        if tmpdir_obj is not None:
            tmpdir_obj.cleanup()

    # -----------------------------------------------------------------
    # Step 7: Render Dockerfile
    # -----------------------------------------------------------------
    spec = {
        "repo": args.repo,
        "base_commit": args.commit,
        "instance_id": instance_id,
        "install_config": install_config,
    }
    dockerfile_content = render_dockerfile(template_path, spec, args.base_image_registry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(dockerfile_content.replace("\r\n", "\n"), encoding="utf-8", newline="\n")
    print(f"Dockerfile written to: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
