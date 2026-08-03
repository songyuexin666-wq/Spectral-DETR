#!/usr/bin/env python3
"""
Batch FPS benchmark runner for the revision experiments.

Runs benchmark_fps.py for each specified checkpoint and aggregates results.

Protocol (per paper Section 4.5):
  - RTX 3090
  - Batch size 1
  - Input resolution 560 × 560
  - 20 warm-up iterations
  - 100 timed iterations
  - Pure model forward only (excludes data loading, pre/post-processing).

Usage:
  python tools/run_fps_benchmarks.py \\
    --checkpoints baseline.pth dafd.pth scu_lue.pth full.pth \\
    --labels baseline dafd scu_lue full \\
    --output fps_results.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


CUDA_GPU_QUERY = (
    "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null"
)


def gpu_info():
    try:
        out = subprocess.check_output(CUDA_GPU_QUERY, shell=True, text=True).strip()
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        if lines:
            return lines[0]
    except Exception:
        pass
    return "unknown"


def run_one(checkpoint: Path, warmup: int, iters: int, device: str):
    """Returns dict with fps, mean_latency_ms, resolution."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve().parent.parent / "benchmark_fps.py"),
        "--checkpoint", str(checkpoint),
        "--device", device,
        "--warmup", str(warmup),
        "--iters", str(iters),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[ERROR] benchmark failed for {checkpoint}")
        print(proc.stderr)
        return {"error": proc.stderr.strip(), "checkpoint": str(checkpoint)}

    # Parse FPS from stdout
    fps = None
    latency_ms = None
    resolution = 560
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("FPS:"):
            try:
                fps = float(line.split(":")[1].strip())
            except Exception:
                pass
        if "Mean latency" in line:
            try:
                latency_ms = float(line.replace("ms / image", "").split(":")[-1].strip())
            except Exception:
                pass
        if line.startswith("Resolution:"):
            try:
                resolution = int(line.split(":")[1].strip().split("x")[0])
            except Exception:
                pass

    return {
        "checkpoint": str(checkpoint),
        "fps": fps,
        "mean_latency_ms": latency_ms,
        "resolution": resolution,
        "warmup_iters": warmup,
        "timed_iters": iters,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", nargs="+", required=True, type=Path,
                        help="Checkpoint paths to benchmark.")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Human-readable labels (same order as checkpoints).")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path.")
    args = parser.parse_args()

    labels = args.labels or [p.stem for p in args.checkpoints]
    if len(labels) != len(args.checkpoints):
        raise SystemExit("Number of labels must match number of checkpoints.")

    gpu = gpu_info()
    results = {
        "gpu": gpu,
        "batch_size": 1,
        "warmup_iters": args.warmup,
        "timed_iters": args.iters,
        "device": args.device,
        "entries": {},
    }

    for cp, label in zip(args.checkpoints, labels):
        if not cp.is_file():
            print(f"[WARN] checkpoint not found, skipping: {cp}")
            continue
        print(f"Benchmarking {label} ({cp}) ...")
        entry = run_one(cp, args.warmup, args.iters, args.device)
        results["entries"][label] = entry

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Print summary
    print(f"\n{'='*60}")
    print(f"GPU: {gpu}")
    print(f"Resolution: 560×560, batch=1, warmup={args.warmup}, iters={args.iters}")
    print(f"{'='*60}")
    for label, entry in results["entries"].items():
        fps = entry.get("fps")
        if fps is not None:
            print(f"  {label:20s}  {fps:6.1f} FPS  ({entry.get('mean_latency_ms', 0):.1f} ms)")
        else:
            print(f"  {label:20s}  ERROR: {entry.get('error', 'unknown')[:80]}")
    print(f"\nResults → {args.output}")


if __name__ == "__main__":
    main()
