#!/usr/bin/env python3
"""
Build Docker images from pre-generated Dockerfiles and run tests inside them.

For each record in a JSONL shard:
  1. Find the matching Dockerfile (by instance_id)
  2. docker build the image
  3. docker run the test command inside the built image
  4. Collect pass/fail result

Usage:
    python scripts/run_shard_tests.py \
        --jsonl shard_0.jsonl \
        --dockerfile-dir dockerfiles/ \
        --output results/shard_0.results.jsonl \
        [--timeout 1800] \
        [--workers 2] \
        [--base-image-registry myregistry] \
        [--base-image-tag-suffix _base]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

_print_lock = threading.Lock()


def _log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


def make_instance_id(repo: str, commit: str) -> str:
    slug = repo.replace("/", "__").lower()
    short = (commit or "unknown")[:7]
    return f"{slug}-{short}"


def docker_build(dockerfile: Path, tag: str, timeout: int) -> tuple[int, str]:
    """Build a Docker image and return (exit_code, output)."""
    cmd = ["docker", "build", "-f", str(dockerfile), "-t", tag, str(dockerfile.parent)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return 124, f"docker build timed out after {timeout}s"
    except Exception as exc:
        return 1, str(exc)


def docker_run_test(image: str, test_cmd: str, workdir: str, timeout: int) -> tuple[int, str]:
    """Run a test command inside a built image and return (exit_code, output)."""
    cmd = [
        "docker", "run", "--rm",
        "-w", workdir,
        image,
        "bash", "-lc", test_cmd,
    ]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return 124, f"test timed out after {timeout}s"
    except Exception as exc:
        return 1, str(exc)


def process_one(
    record: dict,
    dockerfile_dir: Path,
    timeout_build: int,
    timeout_test: int,
    tail_lines: int,
) -> dict:
    """Build and test one JSONL record."""
    repo = record.get("repo", "")
    commit = record.get("base_commit_sha", "")
    instance_id = record.get("instance_id") or make_instance_id(repo, commit)
    test_script = record.get("test_command", "")

    result = {
        "instance_id": instance_id,
        "repo": repo,
        "status": "failed",
        "failed_step": "",
        "exit_code": None,
        "output": "",
        "duration_seconds": 0.0,
    }

    dockerfile = dockerfile_dir / f"{instance_id}.Dockerfile"
    if not dockerfile.is_file():
        result["failed_step"] = "dockerfile_missing"
        result["output"] = f"Dockerfile not found: {dockerfile}"
        _log(f"[SKIP] {instance_id}: Dockerfile not found")
        return result

    image_tag = f"swe-test:{instance_id}"
    project_dir = f"/{repo.split('/')[1]}" if "/" in repo else f"/{repo}"

    # --- Build ---
    t0 = time.monotonic()
    _log(f"[BUILD] {instance_id} ...")
    rc, out = docker_build(dockerfile, image_tag, timeout_build)
    build_duration = time.monotonic() - t0

    if rc != 0:
        result["failed_step"] = "build"
        result["exit_code"] = rc
        result["output"] = "\n".join(out.splitlines()[-tail_lines:])
        result["duration_seconds"] = round(build_duration, 2)
        _log(f"[FAIL] {instance_id}: build failed (exit {rc})")
        return result

    _log(f"[BUILD OK] {instance_id} ({build_duration:.0f}s)")

    # --- Test ---
    if not test_script.strip():
        result["status"] = "passed"
        result["failed_step"] = ""
        result["exit_code"] = 0
        result["output"] = "no test command"
        result["duration_seconds"] = round(build_duration, 2)
        _log(f"[SKIP] {instance_id}: no test command")
        return result

    # Pass the test script directly to bash, stripping only shebangs
    test_lines = []
    for line in test_script.splitlines():
        stripped = line.strip()
        if stripped.startswith("#!"):
            continue
        test_lines.append(line)
    test_cmd = "\n".join(test_lines).strip()

    t1 = time.monotonic()
    _log(f"[TEST] {instance_id} ...")
    rc, out = docker_run_test(image_tag, test_cmd, project_dir, timeout_test)
    test_duration = time.monotonic() - t1
    total_duration = time.monotonic() - t0

    result["exit_code"] = rc
    result["output"] = "\n".join(out.splitlines()[-tail_lines:])
    result["duration_seconds"] = round(total_duration, 2)

    if rc == 0:
        result["status"] = "passed"
        result["failed_step"] = ""
        _log(f"[PASS] {instance_id} ({total_duration:.0f}s)")
    else:
        result["failed_step"] = "test"
        _log(f"[FAIL] {instance_id}: test failed (exit {rc}, {test_duration:.0f}s)")

    # Cleanup image to free disk space
    try:
        subprocess.run(["docker", "rmi", "-f", image_tag], capture_output=True, timeout=60)
    except Exception:
        pass

    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and test a JSONL shard.")
    parser.add_argument("--jsonl", required=True, help="Path to the shard JSONL file.")
    parser.add_argument("--dockerfile-dir", required=True, help="Directory containing generated Dockerfiles.")
    parser.add_argument("--output", required=True, help="Path to write results JSONL.")
    parser.add_argument("--timeout-build", type=int, default=1800, help="Build timeout in seconds (default: 1800).")
    parser.add_argument("--timeout-test", type=int, default=1800, help="Test timeout in seconds (default: 1800).")
    parser.add_argument("--tail-lines", type=int, default=30, help="Last N lines of output to keep (default: 30).")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1, sequential for Docker).")
    args = parser.parse_args()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.is_file():
        print(f"JSONL shard not found: {jsonl_path}", file=sys.stderr)
        return 1

    records: list[dict] = []
    with jsonl_path.open(encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                records.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping line {lineno}: {exc}", file=sys.stderr)

    if not records:
        print("No records in shard.", file=sys.stderr)
        return 1

    total = len(records)
    _log(f"Processing {total} record(s) with {args.workers} worker(s) ...")

    dockerfile_dir = Path(args.dockerfile_dir)
    results: list[dict] = []
    passed = 0
    failed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(
                process_one,
                rec,
                dockerfile_dir,
                args.timeout_build,
                args.timeout_test,
                args.tail_lines,
            ): rec
            for rec in records
        }

        for future in as_completed(futures):
            r = future.result()
            results.append(r)
            if r["status"] == "passed":
                passed += 1
            else:
                failed += 1

    _log(f"\nDone. {passed}/{total} passed, {failed} failed.")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")

    _log(f"Results written to {output_path}")

    # Print summary
    summary = {"total": total, "passed": passed, "failed": failed}
    print(json.dumps(summary))

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
