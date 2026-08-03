#!/usr/bin/env python3
"""
Generate all LaTeX tables for the journal revision from experimental results.

Reads:
  --summary  : aggregate CSV produced by summarize_revision_results.py
  --registry : baseline_registry.yaml (external baselines from literature)
  --fps      : fps_results.json (output of run_fps_benchmarks.py)
  --degradation : controlled-degradation JSON
  --sciencedb   : ScienceDB evaluation JSON

Outputs a self-contained LaTeX file per table under --out-dir.
"""

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Optional

import yaml

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fmt_ap(val, std=None):
    """Format an AP value to exactly 3 decimal places. Returns None when val is missing."""
    if val is None:
        return None
    try:
        fval = float(val)
    except (ValueError, TypeError):
        return str(val)
    if math.isnan(fval):
        return None
    s = f"{fval:.3f}"
    if std is not None:
        try:
            fstd = float(std)
            if not math.isnan(fstd):
                s += f" $\\pm$ {fstd:.3f}"
        except (ValueError, TypeError):
            pass
    return s


def _fmt_int(val):
    if val is None:
        return "---"
    return str(int(val))


def _fmt_float(val, prec=1):
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return "---"
    return f"{float(val):.{prec}f}"


def _bold(text):
    return f"\\textbf{{{text}}}"


def _tt(text):
    return f"\\texttt{{{text}}}"


# ---------------------------------------------------------------------------
# data loaders
# ---------------------------------------------------------------------------


def load_summary(csv_path: Path, fallback: dict = None):
    """Return dict keyed by experiment name.
    Paper-current fallback values are accepted only when the caller explicitly
    opts into draft-placeholder mode.
    """
    rows = {}
    if csv_path and csv_path.exists():
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                rows[row["experiment"]] = {
                    k: float(v) if v else math.nan
                    for k, v in row.items()
                    if k not in ("experiment", "split")
                }
    if fallback:
        for key, vals in fallback.items():
            if key.startswith("_"):
                continue
            if key not in rows:
                rows[key] = {k: float(v) if not isinstance(v, str) else v for k, v in vals.items()}
    return rows


def load_fps(json_path: Path):
    with json_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_degradation(json_path: Path):
    with json_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_sciencedb(json_path: Path):
    with json_path.open(encoding="utf-8") as fh:
        return json.load(fh)


def resolve_entry(entry: dict, summary: dict, fps: dict, sciencedb: dict):
    """Resolve a baseline registry entry that may source from results_json, FPS, or ScienceDB."""
    result = dict(entry)
    source = entry.get("source", "fixed")

    if source == "results_json":
        key = entry.get("key")
        if key and key in summary:
            row = summary[key]
            n = row.get("n", 1)
            # setdefault preserves registry-specified test-set values when
            # the summary only carries validation results (SOTA tables).
            result.setdefault("ap50", _fmt_ap(row.get("ap50_mean"), row.get("ap50_std") if n > 1 else None))
            result.setdefault("ap50_95", _fmt_ap(row.get("ap50_95_mean"), row.get("ap50_95_std") if n > 1 else None))
            result.setdefault("ap_s", _fmt_ap(row.get("ap_s_mean"), row.get("ap_s_std") if n > 1 else None))
            result.setdefault("n_seeds", int(n))
            # Resolve params from checkpoint n_parameters if not already set
            if result.get("params_m") is None:
                pm = _resolve_params(entry, summary)
                if pm is not None:
                    result["params_m"] = pm

    # ScienceDB source: pull values from the ScienceDB eval JSON
    if source == "sciencedb_json" and sciencedb:
        sci_key = entry.get("sciencedb_key", entry.get("key"))
        if isinstance(sciencedb, dict):
            if "ap50" in sciencedb:
                result["ap50"] = _fmt_ap(sciencedb.get("ap50"))
                result["ap50_95"] = _fmt_ap(sciencedb.get("ap50_95"))
                result["ap75"] = _fmt_ap(sciencedb.get("ap75"))
            elif sci_key and sci_key in sciencedb:
                sd = sciencedb[sci_key]
                result["ap50"] = _fmt_ap(sd.get("ap50"))
                result["ap50_95"] = _fmt_ap(sd.get("ap50_95"))
                result["ap75"] = _fmt_ap(sd.get("ap75"))

    # FPS source
    fps_source = entry.get("fps_source")
    if fps_source == "fps_benchmark":
        fps_key = entry.get("fps_key")
        if fps and fps_key in fps:
            result["fps"] = _fmt_float(fps[fps_key].get("fps"), 1)

    # AP source may come from results_json even if the primary source is fps_benchmark
    ap_source = entry.get("ap_source")
    if ap_source == "results_json":
        ap_key = entry.get("ap_key")
        if ap_key and ap_key in summary:
            row = summary[ap_key]
            n = row.get("n", 1)
            result["ap50_95_val"] = _fmt_ap(row.get("ap50_95_mean"), row.get("ap50_95_std") if n > 1 else None)

    # Post-process: ensure all AP-like fields are formatted to 3 decimal places
    # (registry defaults may be raw floats like 0.43).
    for ap_field in ("ap50", "ap50_95", "ap75", "sm_ap50", "sm_ap50_95",
                      "ap_s", "ap50_95_val"):
        raw = result.get(ap_field)
        if raw is not None and not isinstance(raw, str):
            result[ap_field] = _fmt_ap(raw)

    return result


# ---------------------------------------------------------------------------
# table generators
# ---------------------------------------------------------------------------


def _latex_header(label, caption, colspec, font="\\footnotesize"):
    return [
        "\\begin{table}[H]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        font,
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{colspec}}}",
        "\\toprule",
    ]


def _latex_footer():
    return [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]


def _value(entry, metric, bold=False):
    """Return a formatted metric value from an entry dict.

    If *bold* is True and entry['bold'] is truthy, wraps the raw value in
    \\textbf{}. The entry may store the value under *metric* directly or under
    a ``raw_<metric>`` fallback (for labels that carry their own LaTeX markup).
    """
    v = entry.get(metric)
    if v is None:
        v = entry.get(f"raw_{metric}")
    if v is None:
        return "---"
    s = str(v)
    return _bold(s) if (bold and entry.get("bold")) else s


def _label(entry, default=""):
    """Format a row label, handling bold via the `bold` flag rather than inline markup."""
    raw = entry.get("raw_label", default)
    if not raw:
        raw = entry.get("label", default)
    return _bold(raw) if entry.get("bold") else raw


def _resolve_params(entry, summary):
    """Resolve params_m from results.json n_parameters (converted to millions)."""
    if entry.get("params_m") is not None:
        return entry["params_m"]
    key = entry.get("key") or entry.get("ap_key") or entry.get("fps_key")
    if key and key in summary:
        n_params = summary[key].get("n_parameters")
        if n_params is not None and not math.isnan(n_params):
            return round(float(n_params) / 1e6, 1)
    return None


def generate_table_mine_sota(registry, summary, out_dir):
    tbl = registry["mine_objects_sota"]
    rows = []
    for key, entry in tbl["entries"].items():
        resolved = resolve_entry(entry, summary, {}, {})
        rows.append(resolved)

    lines = _latex_header(
        "tab:mine_sota",
        tbl["description"],
        "@{}l c c c c c@{}",
    )
    lines += [
        "& & \\multicolumn{2}{c}{Overall} & \\multicolumn{2}{c}{\\textbf{Small-Object~($^\\dagger$)}} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "Method & Params (M) & AP@0.5 & AP@0.5:0.95 & \\textbf{Sm-AP@0.5} & \\textbf{Sm-AP@0.5:0.95} \\\\",
        "\\midrule",
    ]
    for row in rows:
        b = row.get("bold", False)
        lines.append(
            f"  {_label(row)} & "
            f"{_value(row, 'params_m', b)} & "
            f"{_value(row, 'ap50', b)} & "
            f"{_value(row, 'ap50_95', b)} & "
            f"{_value(row, 'sm_ap50', b)} & "
            f"{_value(row, 'sm_ap50_95', b)} \\\\"
        )
    lines += _latex_footer()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "table_mine_sota.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return lines


def generate_table_exdark_sota(registry, summary, out_dir):
    tbl = registry["exdark_sota"]
    rows = []
    for key, entry in tbl["entries"].items():
        resolved = resolve_entry(entry, summary, {}, {})
        rows.append(resolved)

    lines = _latex_header(
        "tab:exdark",
        tbl["description"],
        "@{}l c c c c c@{}",
    )
    lines += [
        "& & \\multicolumn{2}{c}{Overall} & \\multicolumn{2}{c}{\\textbf{Small-Object~($^\\dagger$)}} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "Method & Params (M) & AP@0.5 & AP@0.5:0.95 & \\textbf{Sm-AP@0.5} & \\textbf{Sm-AP@0.5:0.95} \\\\",
        "\\midrule",
    ]
    for row in rows:
        b = row.get("bold", False)
        lines.append(
            f"  {_label(row)} & "
            f"{_value(row, 'params_m', b)} & "
            f"{_value(row, 'ap50', b)} & "
            f"{_value(row, 'ap50_95', b)} & "
            f"{_value(row, 'sm_ap50', b)} & "
            f"{_value(row, 'sm_ap50_95', b)} \\\\"
        )
    lines += _latex_footer()
    (out_dir / "table_exdark_sota.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_table_sciencedb(registry, summary, sciencedb_results, out_dir):
    tbl = registry["sciencedb_sota"]
    rows = []
    for key, entry in tbl["entries"].items():
        resolved = resolve_entry(entry, summary, {}, sciencedb_results or {})
        rows.append(resolved)

    # Determine if AP@0.75 column is populated
    has_ap75 = any(r.get("ap75") is not None for r in rows)

    colspec = "@{}l l c c c c@{}" if has_ap75 else "@{}l l c c c@{}"
    lines = _latex_header("tab:sciencedb", tbl["description"], colspec, "\\small")

    hdr = "Model & Backbone & Params (M) & AP@0.5 & AP@0.5:0.95"
    if has_ap75:
        hdr += " & AP@0.75"
    hdr += " \\\\"
    lines.append(hdr)
    lines.append("\\midrule")

    for row in rows:
        b = row.get("bold", False)
        backbone_val = _value(row, "backbone", b)
        line = (
            f"  {_label(row)} & "
            f"{backbone_val} & "
            f"{_value(row, 'params_m', b)} & "
            f"{_value(row, 'ap50', b)} & "
            f"{_value(row, 'ap50_95', b)}"
        )
        if has_ap75:
            line += f" & {_value(row, 'ap75', b)}"
        line += " \\\\"
        lines.append(line)

    lines += _latex_footer()
    (out_dir / "table_sciencedb.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_table_ablation(summary, out_dir):
    """Core 8-row ablation (Table 4). All rows come from the core suite."""
    suite_prefix = "mine_objects_core_"
    rows_order = [
        ("baseline", "Baseline (RF-DETR)"),
        ("dafd", "+ DAFD (Stage~1)"),
        ("dqcd", "+ DQCD (Stage~2)"),
        ("scu_lue", "+ SCU+LUE (Stage~3)"),
        ("dafd_dqcd", "+ Stage~1 + 2"),
        ("dafd_scu_lue", "+ Stage~1 + 3"),
        ("dqcd_scu_lue", "+ Stage~2 + 3"),
        ("full", "+ Stage~1+2+3 (Full)"),
    ]

    lines = _latex_header(
        "tab:ablation",
        "Primary component ablation on the Mine-Objects validation set.",
        "@{}l c c c c c c@{}",
        "\\small",
    )
    lines += [
        "Method & AP@0.5:0.95 & AP@0.5 & AP$_S$ & AP$_M$ & Prec. & Recall \\\\",
        "\\midrule",
    ]

    for key, label in rows_order:
        full_key = f"{suite_prefix}{key}"
        row = summary.get(full_key, {})
        n = int(row.get("n", 1))
        is_multi = n > 1

        ap50_95 = _fmt_ap(row.get("ap50_95_mean"), row.get("ap50_95_std") if is_multi else None)
        ap50 = _fmt_ap(row.get("ap50_mean"), row.get("ap50_std") if is_multi else None)
        ap_s = _fmt_ap(row.get("ap_s_mean"), row.get("ap_s_std") if is_multi else None)

        is_full = (key == "full")
        b = lambda x: _bold(x) if is_full else x

        lines.append(
            f"  {b(label) if is_full else label} & "
            f"{b(ap50_95) if is_full else ap50_95} & "
            f"{b(ap50) if is_full else ap50} & "
            f"{b(ap_s) if is_full else ap_s} & "
            f"{'---'} & {'---'} & {'---'} \\\\"
        )

    lines += _latex_footer()
    (out_dir / "table_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_table_coupling(summary, out_dir):
    """Coupling controls: adaptive, fixed, shuffled, random."""
    suite_prefix = "mine_objects_coupling_"
    rows_order = [
        ("full_gate_adaptive", "Adaptive (our design)"),
        ("full_gate_fixed", "Fixed temperature"),
        ("full_gate_shuffled", "Shuffled gate"),
        ("full_gate_random", "Random gate"),
    ]

    lines = _latex_header(
        "tab:coupling",
        "DAFD-to-DQCD coupling controls on Mine-Objects validation.",
        "@{}l c c c@{}",
    )
    lines += [
        "Gate Mode & AP@0.5:0.95 & AP@0.5 & AP$_S$ \\\\",
        "\\midrule",
    ]

    for key, label in rows_order:
        full_key = f"{suite_prefix}{key}"
        row = summary.get(full_key, {})
        ap50_95 = _fmt_ap(row.get("ap50_95_mean"))
        ap50 = _fmt_ap(row.get("ap50_mean"))
        ap_s = _fmt_ap(row.get("ap_s_mean"))
        lines.append(f"  {label} & {ap50_95} & {ap50} & {ap_s} \\\\")

    lines += _latex_footer()
    (out_dir / "table_coupling.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_table_exdark_ablation(summary, out_dir):
    """ExDark external 5-row ablation."""
    suite_prefix = "exdark_external_"
    rows_order = [
        ("baseline", "Baseline (RF-DETR)"),
        ("dafd", "+ DAFD"),
        ("dqcd", "+ DQCD"),
        ("scu_lue", "+ SCU+LUE"),
        ("full", "+ Full (Spectral-DETR)"),
    ]

    lines = _latex_header(
        "tab:exdark_ablation",
        "Component ablation on ExDark validation.",
        "@{}l c c c@{}",
    )
    lines += [
        "Method & AP@0.5:0.95 & AP@0.5 & AP$_S$ \\\\",
        "\\midrule",
    ]

    for key, label in rows_order:
        full_key = f"{suite_prefix}{key}"
        row = summary.get(full_key, {})
        ap50_95 = _fmt_ap(row.get("ap50_95_mean"))
        ap50 = _fmt_ap(row.get("ap50_mean"))
        ap_s = _fmt_ap(row.get("ap_s_mean"))
        b = (key == "full")
        bl = lambda x: _bold(x) if b else x
        lines.append(
            f"  {bl(label) if b else label} & "
            f"{bl(ap50_95) if b else ap50_95} & "
            f"{bl(ap50) if b else ap50} & "
            f"{bl(ap_s) if b else ap_s} \\\\"
        )

    lines += _latex_footer()
    (out_dir / "table_exdark_ablation.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_table_bands(summary, out_dir):
    """DAFD band-count sensitivity: 1–5 bands."""
    suite_prefix = "mine_objects_bands_"
    bands = range(1, 6)

    lines = _latex_header(
        "tab:bands",
        "DAFD band-count sensitivity on Mine-Objects validation.",
        "@{}c c c c@{}",
    )
    lines += [
        "Bands & AP@0.5:0.95 & AP@0.5 & AP$_S$ \\\\",
        "\\midrule",
    ]

    for b in bands:
        key = f"{suite_prefix}dafd_bands_{b}"
        row = summary.get(key, {})
        ap50_95 = _fmt_ap(row.get("ap50_95_mean"))
        ap50 = _fmt_ap(row.get("ap50_mean"))
        ap_s = _fmt_ap(row.get("ap_s_mean"))
        default_marker = " (default)" if b == 3 else ""
        lines.append(f"  {b}{default_marker} & {ap50_95} & {ap50} & {ap_s} \\\\")

    lines += _latex_footer()
    (out_dir / "table_bands.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_table_scu(summary, out_dir):
    """SCU coefficient sensitivity: default + 4 perturbations."""
    suite_prefix = "mine_objects_scu_"
    rows_order = [
        ("scu_default", "Default ($c_1{=}{-}1.5, c_0{=}{-}4.2$)"),
        ("scu_slope_m1_0", "$c_1{=}{-}1.0$"),
        ("scu_slope_m2_0", "$c_1{=}{-}2.0$"),
        ("scu_center_m3_8", "$c_0{=}{-}3.8$"),
        ("scu_center_m4_6", "$c_0{=}{-}4.6$"),
    ]

    lines = _latex_header(
        "tab:scu",
        "SCU coefficient sensitivity on Mine-Objects validation.",
        "@{}l c c c@{}",
    )
    lines += [
        "Setting & AP@0.5:0.95 & AP@0.5 & AP$_S$ \\\\",
        "\\midrule",
    ]

    for key, label in rows_order:
        full_key = f"{suite_prefix}{key}"
        row = summary.get(full_key, {})
        ap50_95 = _fmt_ap(row.get("ap50_95_mean"))
        ap50 = _fmt_ap(row.get("ap50_mean"))
        ap_s = _fmt_ap(row.get("ap_s_mean"))
        lines.append(f"  {label} & {ap50_95} & {ap50} & {ap_s} \\\\")

    lines += _latex_footer()
    (out_dir / "table_scu.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_table_degradation_stratified(summary, degradation_data, out_dir):
    """Table 5: degradation-stratified ΔAP. Source: controlled-degradation eval."""
    lines = _latex_header(
        "tab:deg_stratified",
        "Descriptive degradation-stratified ΔAP@0.5:0.95 over RF-DETR baseline.",
        "@{}l c c c@{}",
    )
    lines += [
        "Degradation Type & Low Severity & Mid Severity & High Severity \\\\",
        "\\midrule",
    ]

    # These values come from the controlled degradation experiment
    # Placeholder structure; filled from degradation_data when available
    if degradation_data:
        metrics = degradation_data.get("settings", [])
        # Group by kind
        from collections import defaultdict

        by_kind = defaultdict(dict)
        for entry in metrics:
            kind = entry.get("kind")
            sev = entry.get("severity")
            ap = entry.get("ap50_95")
            if kind != "clean" and ap is not None:
                by_kind[kind][sev] = ap

        # For each kind, get baseline (clean) and compute Δ
        clean_ap = None
        for entry in metrics:
            if entry.get("kind") == "clean":
                clean_ap = entry.get("ap50_95")
                break

        kind_labels = [
            ("low_light", "Brightness (dark)"),
            ("contrast", "Contrast"),
            ("blur", "Blur"),
        ]
        for kind, label in kind_labels:
            sevs = sorted(by_kind.get(kind, {}).items())
            cells = []
            for _sev, ap in sevs[:3]:
                if clean_ap and ap is not None:
                    delta = ap - clean_ap
                    sign = "+" if delta >= 0 else ""
                    cells.append(f"{sign}{delta:.3f}")
                else:
                    cells.append("---")
            while len(cells) < 3:
                cells.append("---")
            lines.append(f"  {label} & {' & '.join(cells)} \\\\")
    else:
        # Placeholder
        for label in ("Brightness (dark)", "Contrast", "Blur"):
            lines.append(f"  {label} & --- & --- & --- \\\\")

    lines += _latex_footer()
    (out_dir / "table_deg_stratified.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_table_efficiency(registry, summary, fps_data, out_dir):
    """Table 7: efficiency analysis with FPS."""
    tbl = registry["efficiency"]
    rows = []
    for key, entry in tbl["entries"].items():
        resolved = resolve_entry(entry, summary, fps_data or {}, {})
        rows.append(resolved)

    lines = _latex_header(
        "tab:efficiency",
        tbl["description"],
        "@{}l c c c c@{}",
    )
    lines += [
        "Method & Params (M) & FLOPs (G) & FPS & AP@0.5:0.95 \\\\",
        "\\midrule",
    ]

    for row in rows:
        b = row.get("bold", False)
        ap_val = row.get("ap50_95_val") or row.get("ap50_95")
        lines.append(
            f"  {_label(row)} & "
            f"{_value(row, 'params_m', b)} & "
            f"{_value(row, 'flops_g', b)} & "
            f"{_value(row, 'fps', b)} & "
            f"{_bold(ap_val) if (b and row.get('bold')) else (ap_val or '---')} \\\\"
        )

    lines += _latex_footer()
    (out_dir / "table_efficiency.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_table_reliability_evidence(summary, degradation_data, out_dir):
    """Table 6: reliability evidence across stages. Semi-narrative with dynamic values."""
    suite_prefix = "mine_objects_core_"

    # Extract values from summary
    baseline = summary.get(f"{suite_prefix}baseline", {})
    dafd = summary.get(f"{suite_prefix}dafd", {})
    dqcd = summary.get(f"{suite_prefix}dqcd", {})
    scu_lue = summary.get(f"{suite_prefix}scu_lue", {})
    dafd_dqcd = summary.get(f"{suite_prefix}dafd_dqcd", {})

    baseline_ap50 = _fmt_ap(baseline.get("ap50_mean"))
    dafd_ap50 = _fmt_ap(dafd.get("ap50_mean"))
    baseline_ap50_95 = _fmt_ap(baseline.get("ap50_95_mean"))
    dafd_ap50_95 = _fmt_ap(dafd.get("ap50_95_mean"))
    dqcd_ap50_95 = _fmt_ap(dqcd.get("ap50_95_mean"))
    scu_ap50_95 = _fmt_ap(scu_lue.get("ap50_95_mean"))
    baseline_ap_s = _fmt_ap(baseline.get("ap_s_mean"))
    scu_ap_s = _fmt_ap(scu_lue.get("ap_s_mean"))
    baseline_recall = _fmt_ap(baseline.get("recall_mean")) if baseline.get("recall_mean") else "---"
    dafd_dqcd_recall = _fmt_ap(dafd_dqcd.get("recall_mean")) if dafd_dqcd.get("recall_mean") else "---"

    # Correlation from degradation data
    corr = {}
    if degradation_data and "correlations" in degradation_data:
        corr = degradation_data["correlations"]

    lines = _latex_header(
        "tab:reliability_evidence",
        "Reliability evidence across the three stages.",
        "@{}l l c c@{}",
    )
    lines += [
        "Stage & Evidence Metric & Observed Evidence & Source \\\\",
        "\\midrule",
        # DAFD
        f"  \\textbf{{DAFD}} (Feature) & AP@0.5 (standalone) & {baseline_ap50} $\\to$ {dafd_ap50} & Table~\\ref{{tab:ablation}} \\\\",
        "   & Degradation-stratified $\\Delta$AP (blur) & [FILL: deg_blur_high] & Table~\\ref{{tab:deg_stratified}} \\\\",
        "   & Gate activation heatmaps & degradation-dependent & Figure~\\ref{{fig:diagnostics}}a,b \\\\",
        "\\midrule",
        # DQCD
        f"  \\textbf{{DQCD}} (Query) & AP@0.5:0.95 (standalone) & {baseline_ap50_95} $\\to$ {dqcd_ap50_95} & Table~\\ref{{tab:ablation}} \\\\",
        f"   & Stage 1+2 combined Recall & {baseline_recall} $\\to$ {dafd_dqcd_recall} & Table~\\ref{{tab:ablation}} \\\\",
        "   & DQCD temperature ($\\tau$) separation & [FILL: dqcd_temp_sep] & Figure~\\ref{{fig:diagnostics}}c \\\\",
        "\\midrule",
        # SCU+LUE
        f"  \\textbf{{SCU+LUE}} (Local.) & AP$_S$ (standalone) & {baseline_ap_s} $\\to$ {scu_ap_s} & Table~\\ref{{tab:ablation}} \\\\",
        f"   & AP@0.5:0.95 (standalone) & {baseline_ap50_95} $\\to$ {scu_ap50_95} & Table~\\ref{{tab:ablation}} \\\\",
        "   & Error-uncertainty correlation & [FILL: err_uncertainty_corr] & validation diagnostics \\\\",
    ]
    lines += _latex_footer()
    (out_dir / "table_reliability_evidence.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# narrative value map (for fill_paper.py to use)
# ---------------------------------------------------------------------------


def generate_value_map(summary, fps_data, degradation_data, sciencedb_data, paper_fallback, out_dir):
    # Pull diagnostic fallbacks from paper_current when real data absent
    pf = paper_fallback or {}
    """Generate a JSON map of all placeholder keys to their resolved values.
    This is consumed by fill_paper.py and human authors to replace [FILL: key]
    placeholders in the narrative text.
    """
    suite = "mine_objects_core_"
    baseline = summary.get(f"{suite}baseline", {})
    dafd = summary.get(f"{suite}dafd", {})
    dqcd = summary.get(f"{suite}dqcd", {})
    scu_lue = summary.get(f"{suite}scu_lue", {})
    full = summary.get(f"{suite}full", {})

    b_n = int(baseline.get("n", 1))
    f_n = int(full.get("n", 1))

    vm = {
        # Core ablation (validation)
        "baseline_ap50_95": _fmt_ap(baseline.get("ap50_95_mean"), baseline.get("ap50_95_std") if b_n > 1 else None),
        "baseline_ap50": _fmt_ap(baseline.get("ap50_mean"), baseline.get("ap50_std") if b_n > 1 else None),
        "baseline_ap_s": _fmt_ap(baseline.get("ap_s_mean"), baseline.get("ap_s_std") if b_n > 1 else None),
        "dafd_ap50_95": _fmt_ap(dafd.get("ap50_95_mean")),
        "dafd_ap50": _fmt_ap(dafd.get("ap50_mean")),
        "dafd_ap_s": _fmt_ap(dafd.get("ap_s_mean")),
        "dqcd_ap50_95": _fmt_ap(dqcd.get("ap50_95_mean")),
        "dqcd_ap50": _fmt_ap(dqcd.get("ap50_mean")),
        "scu_ap50_95": _fmt_ap(scu_lue.get("ap50_95_mean")),
        "scu_ap50": _fmt_ap(scu_lue.get("ap50_mean")),
        "scu_ap_s": _fmt_ap(scu_lue.get("ap_s_mean")),
        "full_ap50_95": _fmt_ap(full.get("ap50_95_mean"), full.get("ap50_95_std") if f_n > 1 else None),
        "full_ap50": _fmt_ap(full.get("ap50_mean"), full.get("ap50_std") if f_n > 1 else None),
        "full_ap_s": _fmt_ap(full.get("ap_s_mean"), full.get("ap_s_std") if f_n > 1 else None),
        "full_n_seeds": f_n,
    }

    # Compute individual gains
    for metric, label in [("ap50_95", "AP@0.5:0.95"), ("ap50", "AP@0.5"), ("ap_s", "AP_S")]:
        bv = baseline.get(f"{metric}_mean", 0)
        fv = full.get(f"{metric}_mean", 0)
        if bv and fv:
            vm[f"gain_{metric}"] = f"{fv - bv:+.3f}"
        else:
            vm[f"gain_{metric}"] = "[TO_BE_MEASURED]"

    # Interaction: full gain minus sum of individual gains
    for metric in ["ap50_95", "ap50"]:
        bv = baseline.get(f"{metric}_mean")
        dv = dafd.get(f"{metric}_mean")
        qv = dqcd.get(f"{metric}_mean")
        sv = scu_lue.get(f"{metric}_mean")
        fv = full.get(f"{metric}_mean")
        if all(x is not None and not math.isnan(x) for x in [bv, dv, qv, sv, fv]):
            total_gain = fv - bv
            sum_individual = (dv - bv) + (qv - bv) + (sv - bv)
            interaction = total_gain - sum_individual
            vm[f"interaction_{metric}"] = f"{interaction:+.3f}"
        else:
            vm[f"interaction_{metric}"] = "[TO_BE_MEASURED]"

    # FPS
    for label, key, fallback_key in [
        ("baseline_fps", "mine_objects_core_baseline", "_diag_baseline_fps"),
        ("dafd_fps", "mine_objects_core_dafd", "_diag_dafd_fps"),
        ("scu_lue_fps", "mine_objects_core_scu_lue", "_diag_scu_lue_fps"),
        ("full_fps", "mine_objects_core_full", "_diag_full_fps"),
    ]:
        if fps_data and key in fps_data:
            vm[label] = _fmt_float(fps_data[key].get("fps"), 1)
            vm[f"{label}_ms"] = _fmt_float(fps_data[key].get("mean_latency_ms"), 1)
        elif pf.get(fallback_key) is not None:
            vm[label] = _fmt_float(pf[fallback_key], 1)
        else:
            vm[label] = "[TO_BE_MEASURED]"

    # Full test-set results (from results.json, not validation)
    vm["full_test_ap50"] = "[TO_BE_MEASURED]"
    vm["full_test_ap50_95"] = "[TO_BE_MEASURED]"

    # ScienceDB
    if sciencedb_data and sciencedb_data.get("ap50") is not None:
        for met in ("ap50", "ap50_95", "ap75"):
            vm[f"sciencedb_{met}"] = sciencedb_data.get(met, "[TO_BE_MEASURED]")
        gap = (sciencedb_data.get("ap50", 0) or 0) - (sciencedb_data.get("ap50_95", 0) or 0)
        vm["sciencedb_ap_gap"] = f"{gap:.3f}"
    else:
        for met in ("ap50", "ap50_95", "ap75"):
            fbk = pf.get(f"_diag_sciencedb_{met}")
            vm[f"sciencedb_{met}"] = _fmt_ap(fbk) if fbk is not None else "[TO_BE_MEASURED]"
        vm["sciencedb_ap_gap"] = pf.get("_diag_sciencedb_ap_gap", "[TO_BE_MEASURED]")

    # ExDark test results
    vm["exdark_test_ap50"] = "[TO_BE_MEASURED]"
    vm["exdark_test_ap50_95"] = "[TO_BE_MEASURED]"

    # Coupling deltas: adaptive vs shuffled/random
    coupling_prefix = "mine_objects_coupling_"
    for mode in ("adaptive", "fixed", "shuffled", "random"):
        key = f"{coupling_prefix}full_gate_{mode}"
        row = summary.get(key, {})
        vm[f"coupling_{mode}_ap50_95"] = _fmt_ap(row.get("ap50_95_mean"))

    # Diagnostics
    if degradation_data and "correlations" in degradation_data:
        corr = degradation_data["correlations"]
        for kind in ("low_light", "noise", "blur", "contrast"):
            c = corr.get(kind, {})
            for pair in ("severity_vs_gate", "gate_vs_localization_error"):
                cpair = c.get(pair, {})
                if cpair:
                    vm[f"corr_{kind}_{pair}_spearman"] = _fmt_float(cpair.get("spearman"), 3)
                    vm[f"corr_{kind}_{pair}_pearson"] = _fmt_float(cpair.get("pearson"), 3)
        # Uncertainty-error overall
        # Use severity_vs_localization_error across all degradation kinds for overall
        all_gate = []
        all_err = []
        for setting in degradation_data.get("settings", []):
            records = setting.get("records", [])
            for r in records:
                if r.get("gate_mean") is not None and r.get("localization_error") is not None:
                    all_gate.append(r["gate_mean"])
                    all_err.append(r["localization_error"])
        if len(all_gate) >= 3:
            import numpy as np
            from scipy import stats as scipy_stats
            pearson = float(np.corrcoef(all_gate, all_err)[0, 1])
            spearman = float(scipy_stats.spearmanr(all_gate, all_err).statistic)
            vm["uncertainty_error_pearson"] = f"{pearson:.2f}"
            vm["uncertainty_error_spearman"] = f"{spearman:.2f}"
        else:
            vm["uncertainty_error_pearson"] = pf.get("_diag_uncertainty_error_pearson", "[TO_BE_MEASURED]")
            vm["uncertainty_error_spearman"] = pf.get("_diag_uncertainty_error_spearman", "[TO_BE_MEASURED]")

        # Risk-coverage
        rc = degradation_data.get("risk_coverage", {})
        if rc:
            for cov, val in rc.items():
                vm[f"risk_coverage_{cov}"] = _fmt_float(val, 3)
        else:
            for cov in ("0.25", "0.5", "0.75", "1.0"):
                fbk = pf.get(f"_diag_risk_coverage_{cov}")
                vm[f"risk_coverage_{cov}"] = _fmt_float(fbk, 3) if fbk is not None else "[TO_BE_MEASURED]"
    else:
        vm["uncertainty_error_pearson"] = pf.get("_diag_uncertainty_error_pearson", "[TO_BE_MEASURED]")
        vm["uncertainty_error_spearman"] = pf.get("_diag_uncertainty_error_spearman", "[TO_BE_MEASURED]")
        for cov in ("0.25", "0.5", "0.75", "1.0"):
            fbk = pf.get(f"_diag_risk_coverage_{cov}")
            vm[f"risk_coverage_{cov}"] = _fmt_float(fbk, 3) if fbk is not None else "[TO_BE_MEASURED]"

    # Bands
    bands_prefix = "mine_objects_bands_dafd_bands_"
    for b in range(1, 6):
        row = summary.get(f"{bands_prefix}{b}", {})
        vm[f"bands_{b}_ap50_95"] = _fmt_ap(row.get("ap50_95_mean"))

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "value_map.json").write_text(
        json.dumps(vm, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return vm


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, help="Path to summary CSV from summarize_revision_results.py")
    parser.add_argument("--registry", type=Path, default=Path(__file__).parent / "baseline_registry.yaml")
    parser.add_argument("--fps", type=Path, help="Path to fps_results.json")
    parser.add_argument("--degradation", type=Path, help="Path to controlled-degradation output JSON")
    parser.add_argument("--sciencedb", type=Path, help="Path to ScienceDB evaluation JSON")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output directory for LaTeX tables")
    parser.add_argument("--paper-fallback", type=Path,
                        default=Path(__file__).parent / "paper_current.yaml",
                        help="YAML with paper-current values used when results.json is absent")
    parser.add_argument(
        "--allow-paper-fallback", action="store_true",
        help="Explicitly generate DRAFT PLACEHOLDER tables from paper-current values",
    )
    parser.add_argument("--tables", nargs="*",
                        choices=["all", "sota_mine", "sota_exdark", "sota_sciencedb",
                                  "ablation", "coupling", "exdark_ablation", "bands", "scu",
                                  "deg_stratified", "efficiency", "reliability"],
                        default=["all"],
                        help="Which tables to generate")
    args = parser.parse_args()

    registry = yaml.safe_load(args.registry.read_text(encoding="utf-8")) if args.registry.exists() else {}
    if not args.summary or not args.summary.exists():
        if not args.allow_paper_fallback:
            parser.error("--summary must point to a real aggregation CSV; use --allow-paper-fallback only for visibly marked drafts")
    paper_fallback = {}
    if args.allow_paper_fallback:
        if not args.paper_fallback or not args.paper_fallback.exists():
            parser.error("--allow-paper-fallback requires an existing --paper-fallback YAML")
        paper_fallback = yaml.safe_load(args.paper_fallback.read_text(encoding="utf-8")) or {}
    summary = load_summary(args.summary, paper_fallback)
    fps_data = load_fps(args.fps) if args.fps and args.fps.exists() else {}
    degradation_data = load_degradation(args.degradation) if args.degradation and args.degradation.exists() else {}
    sciencedb_data = load_sciencedb(args.sciencedb) if args.sciencedb and args.sciencedb.exists() else {}

    run_all = "all" in args.tables

    if run_all or "sota_mine" in args.tables:
        generate_table_mine_sota(registry, summary, args.out_dir)
    if run_all or "sota_exdark" in args.tables:
        generate_table_exdark_sota(registry, summary, args.out_dir)
    if run_all or "sota_sciencedb" in args.tables:
        generate_table_sciencedb(registry, summary, sciencedb_data, args.out_dir)
    if run_all or "ablation" in args.tables:
        generate_table_ablation(summary, args.out_dir)
    if run_all or "coupling" in args.tables:
        generate_table_coupling(summary, args.out_dir)
    if run_all or "exdark_ablation" in args.tables:
        generate_table_exdark_ablation(summary, args.out_dir)
    if run_all or "bands" in args.tables:
        generate_table_bands(summary, args.out_dir)
    if run_all or "scu" in args.tables:
        generate_table_scu(summary, args.out_dir)
    if run_all or "deg_stratified" in args.tables:
        generate_table_degradation_stratified(summary, degradation_data, args.out_dir)
    if run_all or "efficiency" in args.tables:
        generate_table_efficiency(registry, summary, fps_data, args.out_dir)
    if run_all or "reliability" in args.tables:
        generate_table_reliability_evidence(summary, degradation_data, args.out_dir)

    # Always generate the value map
    generate_value_map(summary, fps_data, degradation_data, sciencedb_data, paper_fallback, args.out_dir)

    provenance = {
        "mode": "draft_placeholder" if args.allow_paper_fallback else "measured_results",
        "summary": str(args.summary.resolve()) if args.summary and args.summary.exists() else None,
        "paper_fallback": str(args.paper_fallback.resolve()) if args.allow_paper_fallback else None,
        "fps": str(args.fps.resolve()) if args.fps and args.fps.exists() else None,
        "degradation": str(args.degradation.resolve()) if args.degradation and args.degradation.exists() else None,
        "sciencedb": str(args.sciencedb.resolve()) if args.sciencedb and args.sciencedb.exists() else None,
    }
    (args.out_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2), encoding="utf-8"
    )
    if args.allow_paper_fallback:
        for table_path in args.out_dir.glob("table_*.tex"):
            content = table_path.read_text(encoding="utf-8")
            content = "% DRAFT PLACEHOLDER: NOT MEASURED EVIDENCE\n" + content.replace(
                "\\caption{", "\\caption{DRAFT PLACEHOLDER: ", 1
            )
            table_path.write_text(content, encoding="utf-8")

    print(f"Tables written to {args.out_dir}/")
    for f in sorted(args.out_dir.glob("table_*.tex")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
