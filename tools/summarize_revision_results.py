#!/usr/bin/env python3
"""Aggregate selected-checkpoint metrics from revision experiment results."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = {"ap50_95": 0, "ap50": 1, "ap_s": 3}


def infer_run_metadata(path):
    config_path = path.parent / "config.yaml"
    name = path.parent.name
    seed = None
    if config_path.exists():
        try:
            import yaml

            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            name = config.get("experiment_name", name)
            seed = config.get("training", {}).get("seed")
        except Exception:
            pass
    if seed is None and "_seed" in name:
        tail = name.rsplit("_seed", 1)[1]
        if tail.isdigit():
            seed = int(tail)
    return name, seed


def load_result(path, split):
    payload = json.loads(path.read_text(encoding="utf-8"))
    coco = payload.get("coco_eval_bbox")
    if isinstance(coco, dict):
        values = coco.get(split)
    else:
        values = coco if split == "valid" else None
    if not values or len(values) < 4:
        return None
    name, seed = infer_run_metadata(path)
    row = {
        "run": name,
        "seed": seed,
        "split": split,
        "results_path": str(path.resolve()),
        "selected_checkpoint": payload.get("selected_checkpoint"),
        "selected_checkpoint_source": payload.get("selected_checkpoint_source"),
    }
    row.update({metric: float(values[index]) for metric, index in METRICS.items()})
    # Forward n_parameters for params/footprint reporting
    n_params = payload.get("n_parameters")
    if n_params is not None:
        row["n_parameters"] = int(n_params)
    return row


def group_name(run):
    return run.rsplit("_seed", 1)[0] if "_seed" in run else run


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[(group_name(row["run"]), row["split"])].append(row)
    summary = []
    for (name, split), items in sorted(groups.items()):
        record = {"experiment": name, "split": split, "n": len(items)}
        for metric in METRICS:
            values = [item[metric] for item in items]
            record[f"{metric}_mean"] = statistics.fmean(values)
            record[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else math.nan
        # Forward n_parameters from the first item (same config = same params)
        n_params = items[0].get("n_parameters")
        if n_params is not None:
            record["n_parameters"] = n_params
        summary.append(record)
    return summary


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Build fieldnames from first row plus common extras
    base_fields = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=base_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, rows):
    lines = [
        "| Experiment | Split | n | AP@0.5:0.95 | AP@0.5 | AP_S |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def format_metric(metric):
            mean = row[f"{metric}_mean"]
            std = row[f"{metric}_std"]
            return f"{mean:.3f}" if math.isnan(std) else f"{mean:.3f} +/- {std:.3f}"

        lines.append(
            f"| {row['experiment']} | {row['split']} | {row['n']} | "
            f"{format_metric('ap50_95')} | {format_metric('ap50')} | "
            f"{format_metric('ap_s')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--split", choices=("valid", "test"), default="valid")
    parser.add_argument("--output-prefix", required=True, type=Path)
    args = parser.parse_args()

    rows = []
    for path in sorted(args.root.rglob("results.json")):
        row = load_result(path, args.split)
        if row is not None:
            rows.append(row)
    if not rows:
        raise SystemExit(f"No usable {args.split} results.json files under {args.root}")

    summary = aggregate(rows)
    write_csv(args.output_prefix.with_suffix(".runs.csv"), rows)
    write_csv(args.output_prefix.with_suffix(".summary.csv"), summary)
    write_markdown(args.output_prefix.with_suffix(".md"), summary)
    print(f"Summarized {len(rows)} runs into {len(summary)} experiment rows")


if __name__ == "__main__":
    main()
