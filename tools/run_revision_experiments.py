#!/usr/bin/env python3
"""Generate or execute the compact experiment matrix for the journal revision."""

import argparse
import copy
import hashlib
import json
import platform
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


# Rows that receive the extra seeds requested for mean/variance reporting. Every
# other row runs once at the primary seed, which keeps the training budget at
# 12 runs for the core suite instead of 24.
MULTI_SEED_EXPERIMENTS = {"baseline", "full"}

MODULE_DEFAULTS = {
    "use_dafd": False,
    "use_dqcd": False,
    "use_lue": False,
    "use_scu": False,
}


def deep_update(target, updates):
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def module_overrides(*, dafd=False, dqcd=False, lue=False, scu=False):
    model = dict(MODULE_DEFAULTS)
    model.update(
        use_dafd=dafd,
        use_dqcd=dqcd,
        use_lue=lue,
        use_scu=scu,
    )
    return {"model": model}


def experiment_matrix(suite):
    if suite == "core":
        return [
            ("baseline", module_overrides()),
            ("dafd", module_overrides(dafd=True)),
            ("dqcd", module_overrides(dqcd=True)),
            ("scu_lue", module_overrides(lue=True, scu=True)),
            ("dafd_dqcd", module_overrides(dafd=True, dqcd=True)),
            ("dafd_scu_lue", module_overrides(dafd=True, lue=True, scu=True)),
            ("dqcd_scu_lue", module_overrides(dqcd=True, lue=True, scu=True)),
            ("full", module_overrides(dafd=True, dqcd=True, lue=True, scu=True)),
        ]
    if suite == "coupling":
        return [
            (
                f"full_gate_{mode}",
                deep_update(
                    module_overrides(dafd=True, dqcd=True, lue=True, scu=True),
                    {"model": {"dqcd_gate_mode": mode}},
                ),
            )
            for mode in ("adaptive", "fixed", "shuffled", "random")
        ]
    if suite == "external":
        return [
            ("baseline", module_overrides()),
            ("dafd", module_overrides(dafd=True)),
            ("dqcd", module_overrides(dqcd=True)),
            ("scu_lue", module_overrides(lue=True, scu=True)),
            ("full", module_overrides(dafd=True, dqcd=True, lue=True, scu=True)),
        ]
    if suite == "bands":
        return [
            (
                f"dafd_bands_{bands}",
                deep_update(
                    module_overrides(dafd=True),
                    {"model": {"dafd_n_bands": bands}},
                ),
            )
            for bands in range(1, 6)
        ]
    if suite == "scu":
        settings = [
            ("default", -1.5, -4.2),
            ("slope_m1_0", -1.0, -4.2),
            ("slope_m2_0", -2.0, -4.2),
            ("center_m3_8", -1.5, -3.8),
            ("center_m4_6", -1.5, -4.6),
        ]
        return [
            (
                f"scu_{name}",
                deep_update(
                    module_overrides(lue=True, scu=True),
                    {
                        "model": {
                            "scu_calib_slope": slope,
                            "scu_calib_center": center,
                        }
                    },
                ),
            )
            for name, slope, center in settings
        ]
    if suite == "taps":
        rows = []
        for tap in range(4):
            rows.append(
                (
                    f"dafd_tap_{tap}",
                    deep_update(
                        module_overrides(dafd=True),
                        {
                            "model": {
                                "dafd_feature_indices": [tap],
                                "dafd_gate_source_index": tap,
                            }
                        },
                    ),
                )
            )
        rows.append(
            (
                "dafd_all_taps",
                deep_update(
                    module_overrides(dafd=True),
                    {
                        "model": {
                            "dafd_feature_indices": [0, 1, 2, 3],
                            "dafd_gate_source_index": 0,
                        }
                    },
                ),
            )
        )
        return rows
    raise ValueError(f"Unknown suite: {suite}")


def seeds_for(experiment, primary_seed, extra_seeds):
    """Primary seed for every row; extra seeds only where variance is reported."""
    if experiment in MULTI_SEED_EXPERIMENTS:
        return [primary_seed, *extra_seeds]
    return [primary_seed]


def config_digest(config):
    payload = json.dumps(config, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def git_provenance():
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", *args], stderr=subprocess.DEVNULL, text=True
            ).strip()
        except Exception:
            return None

    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


def is_complete(run_dir):
    """A run counts as finished only if the selected checkpoint was re-evaluated."""
    results = Path(run_dir) / "results.json"
    if not results.exists():
        return False
    try:
        payload = json.loads(results.read_text(encoding="utf-8"))
    except Exception:
        return False
    coco = payload.get("coco_eval_bbox")
    values = coco.get("valid") if isinstance(coco, dict) else coco
    return bool(values) and payload.get("selected_checkpoint") is not None


def build_runs(base, suites, dataset_name, dataset_path, primary_seed, extra_seeds, output_root):
    runs = []
    for suite in suites:
        for experiment, overrides in experiment_matrix(suite):
            for seed in seeds_for(experiment, primary_seed, extra_seeds):
                run_name = f"{dataset_name}_{suite}_{experiment}_seed{seed}"
                config = deep_update(copy.deepcopy(base), overrides)
                config["experiment_name"] = run_name
                config.setdefault("dataset", {})["dataset_file"] = "coco"
                config["dataset"]["coco_path"] = str(dataset_path)
                config.setdefault("training", {})["seed"] = seed
                # Deterministic run directory: required so every results.json maps
                # back to exactly one manifest row.
                config["training"]["timestamp_output_dir"] = False
                config["training"]["output_dir"] = str(output_root / "runs" / run_name)
                runs.append(
                    {
                        "suite": suite,
                        "experiment": experiment,
                        "dataset": dataset_name,
                        "seed": seed,
                        "name": run_name,
                        "output_dir": config["training"]["output_dir"],
                        "config_digest": config_digest(config),
                        "config": config,
                    }
                )
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument(
        "--suite",
        action="append",
        choices=("core", "coupling", "external", "bands", "scu", "taps"),
        required=True,
        help="Repeat to generate multiple suites.",
    )
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--dataset-path", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Primary seed used by every row.",
    )
    parser.add_argument(
        "--extra-seed",
        action="append",
        type=int,
        default=None,
        help=(
            "Repeatable. Applied only to the baseline and full rows, which are the "
            "rows reported with mean and standard deviation."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs whose output directory already holds a complete results.json.",
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    base = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    runs = build_runs(
        base,
        args.suite,
        args.dataset_name,
        args.dataset_path.resolve(),
        args.seed,
        args.extra_seed or [],
        args.output_root.resolve(),
    )

    config_dir = args.output_root / "generated_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    provenance = git_provenance()
    manifest = []
    executed = skipped = 0
    for run in runs:
        config_path = config_dir / f"{run['name']}.yaml"
        config_path.write_text(
            yaml.safe_dump(run["config"], sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        command = [args.python, "train_mine.py", "--config", str(config_path)]
        complete = is_complete(run["output_dir"])
        manifest.append(
            {key: run[key] for key in run if key != "config"}
            | {
                "config_path": str(config_path),
                "command": command,
                "provenance": provenance,
                "complete_at_generation": complete,
            }
        )
        if args.resume and complete:
            print(f"# skip (complete): {run['name']}")
            skipped += 1
            continue
        print(shlex.join(command))
        if args.execute:
            subprocess.run(command, check=True)
            executed += 1

    manifest_path = args.output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    action = "Executed" if args.execute else "Generated"
    print(
        f"{action} {executed if args.execute else len(runs) - skipped} runs "
        f"({skipped} skipped as complete); manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
