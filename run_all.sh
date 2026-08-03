#!/usr/bin/env bash
# ============================================================================
# Spectral-DETR Journal Revision — Master Experiment Script
# ============================================================================
#
# Run this ONCE on your AutoDL instance (Python 3.10+, RTX 3090, 24 GB).
# It executes the complete revision experiment protocol in priority order,
# skipping any step whose output already exists (resume-safe).
#
# Usage:
#   1. Edit the PATHS section below to match your AutoDL filesystem.
#   2. chmod +x run_all.sh
#   3. bash run_all.sh 2>&1 | tee run_all.log
#
# After completion:
#   results/paper_tables/     ← all LaTeX tables ready for \input
#   results/paper_value_map.json  ← all placeholder → value mappings
#   results/summary/          ← aggregated CSVs
#   results/figures/          ← Figure 5 panels
#   results/diagnostics/      ← controlled degradation & risk-coverage
# ============================================================================

set -euo pipefail

# ── PATHS ──────────────────────────────────────────────────────────────────
# EDIT THESE to match your AutoDL setup.

MINE_OBJECTS_PATH="/root/autodl-tmp/mine_objects"      # COCO directory (images + _annotations.coco.json)
EXDARK_PATH="/root/autodl-tmp/exdark"                   # COCO directory
SCIENCEDB_PATH="/root/autodl-tmp/sciencedb"             # COCO directory
SCIENCEDB_CHECKPOINT=""                                 # dedicated 5-class checkpoint
BASE_CONFIG="configs/ablation_softnms_dafd_dqcd_lue_scu.yaml"
OUTPUT_ROOT="revision_runs"                             # output root (can be relative)
PRIMARY_SEED=42
EXTRA_SEEDS=(123 456)                                   # for baseline/full variance
FIG5_IMAGES=""                                          # space-separated list of image paths for Fig 5

# ── DERIVED ────────────────────────────────────────────────────────────────

SCRIPTS="tools/run_revision_experiments.py"
SUMMARIZE="tools/summarize_revision_results.py"
MAKE_TABLES="tools/make_tables.py"
FPS_RUNNER="tools/run_fps_benchmarks.py"
SCIENCEDB_EVAL="tools/eval_sciencedb.py"
DEGRADATION_EVAL="tools/evaluate_controlled_degradation.py"
FIG5_EXPORT="tools/export_figure5.py"
PAPER_TABLES="results/paper_tables"
SUMMARY_DIR="results/summary"
FIGURES_DIR="results/figures"
DIAG_DIR="results/diagnostics"
PYTHON="${PYTHON:-python}"

mkdir -p "$SUMMARY_DIR" "$PAPER_TABLES" "$FIGURES_DIR" "$DIAG_DIR"

# ── HELPERS ────────────────────────────────────────────────────────────────

banner() {
    echo ""
    echo "================================================================================"
    echo "  $*"
    echo "================================================================================"
    echo ""
}

check_python() {
    banner "Checking environment"
    $PYTHON --version
    $PYTHON -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')"
    if [ "$(nvidia-smi -L 2>/dev/null | wc -l)" -gt 0 ]; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        echo "[WARN] No NVIDIA GPU visible."
    fi
}

# ── STEP 1 ─────────────────────────────────────────────────────────────────

run_core() {
    banner "STEP 1/8 — Mine-Objects Core Ablation (8 rows × seed strategy)"
    local MANIFEST="$OUTPUT_ROOT/mine_objects/manifest.json"
    if [ -f "$MANIFEST" ]; then
        echo "[SKIP] Manifest exists. Resume mode: skipping completed runs only."
    fi
    $PYTHON "$SCRIPTS" \
        --base-config "$BASE_CONFIG" \
        --suite core \
        --dataset-name mine_objects \
        --dataset-path "$MINE_OBJECTS_PATH" \
        --output-root "$OUTPUT_ROOT/mine_objects" \
        --seed "$PRIMARY_SEED" \
        $(for s in "${EXTRA_SEEDS[@]}"; do echo "--extra-seed $s"; done) \
        --resume \
        --execute
}

# ── STEP 2 ─────────────────────────────────────────────────────────────────

run_external() {
    banner "STEP 2/8 — ExDark External Ablation (5 rows)"
    $PYTHON "$SCRIPTS" \
        --base-config "$BASE_CONFIG" \
        --suite external \
        --dataset-name exdark \
        --dataset-path "$EXDARK_PATH" \
        --output-root "$OUTPUT_ROOT/exdark" \
        --seed "$PRIMARY_SEED" \
        --resume \
        --execute
}

# ── STEP 3 ─────────────────────────────────────────────────────────────────

run_coupling() {
    banner "STEP 3/8 — DAFD→DQCD Coupling Controls (4 rows)"
    $PYTHON "$SCRIPTS" \
        --base-config "$BASE_CONFIG" \
        --suite coupling \
        --dataset-name mine_objects \
        --dataset-path "$MINE_OBJECTS_PATH" \
        --output-root "$OUTPUT_ROOT/coupling" \
        --seed "$PRIMARY_SEED" \
        --resume \
        --execute
}

# ── STEP 4 ─────────────────────────────────────────────────────────────────

run_bands() {
    banner "STEP 4/8 — DAFD Band-Count Sensitivity (5 rows: 1–5 bands)"
    $PYTHON "$SCRIPTS" \
        --base-config "$BASE_CONFIG" \
        --suite bands \
        --dataset-name mine_objects \
        --dataset-path "$MINE_OBJECTS_PATH" \
        --output-root "$OUTPUT_ROOT/bands" \
        --seed "$PRIMARY_SEED" \
        --resume \
        --execute
}

# ── STEP 5 ─────────────────────────────────────────────────────────────────

run_scu() {
    banner "STEP 5/8 — SCU Coefficient Sensitivity (5 rows)"
    $PYTHON "$SCRIPTS" \
        --base-config "$BASE_CONFIG" \
        --suite scu \
        --dataset-name mine_objects \
        --dataset-path "$MINE_OBJECTS_PATH" \
        --output-root "$OUTPUT_ROOT/scu" \
        --seed "$PRIMARY_SEED" \
        --resume \
        --execute
}

# ── STEP 6 ─────────────────────────────────────────────────────────────────

run_fps() {
    banner "STEP 6/8 — FPS Benchmark (baseline, DAFD, SCU+LUE, Full)"
    local CKPT_DIR="$OUTPUT_ROOT/mine_objects/runs"
    local BASELINE_CKPT="$CKPT_DIR/mine_objects_core_baseline_seed${PRIMARY_SEED}/checkpoint_best_total.pth"
    local DAFD_CKPT="$CKPT_DIR/mine_objects_core_dafd_seed${PRIMARY_SEED}/checkpoint_best_total.pth"
    local SCU_CKPT="$CKPT_DIR/mine_objects_core_scu_lue_seed${PRIMARY_SEED}/checkpoint_best_total.pth"
    local FULL_CKPT="$CKPT_DIR/mine_objects_core_full_seed${PRIMARY_SEED}/checkpoint_best_total.pth"
    local FPS_OUT="$DIAG_DIR/fps_results.json"

    if [ -f "$FPS_OUT" ]; then
        echo "[SKIP] FPS results already exist at $FPS_OUT"
        return
    fi

    # Collect existing checkpoints
    local CMD=(--checkpoints)
    local LBL=(--labels)
    for pair in "$BASELINE_CKPT:baseline" "$DAFD_CKPT:dafd" "$SCU_CKPT:scu_lue" "$FULL_CKPT:full"; do
        local cp="${pair%%:*}" lb="${pair##*:}"
        if [ -f "$cp" ]; then
            CMD+=("$cp"); LBL+=("$lb")
        else
            echo "[WARN] Checkpoint not found: $cp"
        fi
    done

    if [ ${#CMD[@]} -le 2 ]; then
        echo "[ERROR] No checkpoints found for FPS benchmark. Run steps 1-5 first."
        return 1
    fi

    $PYTHON "$FPS_RUNNER" "${CMD[@]}" "${LBL[@]}" --output "$FPS_OUT"
}

# ── STEP 7 ─────────────────────────────────────────────────────────────────

run_fig5() {
    banner "STEP 7/8 — Export Figure 5 (qualitative comparison)"
    if [ -z "$FIG5_IMAGES" ]; then
        echo "[SKIP] No FIG5_IMAGES configured. Edit the script to set image paths."
        return
    fi
    local CKPT_DIR="$OUTPUT_ROOT/mine_objects/runs"
    local BASELINE_CKPT="$CKPT_DIR/mine_objects_core_baseline_seed${PRIMARY_SEED}/checkpoint_best_total.pth"
    local FULL_CKPT="$CKPT_DIR/mine_objects_core_full_seed${PRIMARY_SEED}/checkpoint_best_total.pth"

    if [ -f "$FIGURES_DIR/fig5_manifest.json" ]; then
        echo "[SKIP] Figure 5 already exported."
        return
    fi

    $PYTHON "$FIG5_EXPORT" \
        --baseline "$BASELINE_CKPT" --full "$FULL_CKPT" \
        --images $FIG5_IMAGES \
        --score-thresh 0.3 \
        --num-classes 14 \
        --output-dir "$FIGURES_DIR"
}

# ── STEP 8 ─────────────────────────────────────────────────────────────────

run_diagnostics() {
    banner "STEP 8/8 — Diagnostics: degradation, ScienceDB, tables"

    # 8a — Controlled degradation
    local CKPT_DIR="$OUTPUT_ROOT/mine_objects/runs"
    local FULL_CKPT="$CKPT_DIR/mine_objects_core_full_seed${PRIMARY_SEED}/checkpoint_best_total.pth"
    local DEGRADATION_OUT="$DIAG_DIR/controlled_degradation.json"

    if [ -f "$FULL_CKPT" ] && [ ! -f "$DEGRADATION_OUT" ]; then
        echo "--- 8a: Controlled degradation diagnostics ---"
        $PYTHON "$DEGRADATION_EVAL" \
            --checkpoint "$FULL_CKPT" \
            --coco-path "$MINE_OBJECTS_PATH" \
            --output "$DEGRADATION_OUT" \
            --device cuda
    else
        echo "[SKIP] Controlled degradation: output exists or checkpoint missing."
    fi

    # 8b — ScienceDB evaluation (multi-IoU)
    local SCIENCEDB_OUT="$DIAG_DIR/sciencedb_results.json"
    if [ -n "$SCIENCEDB_CHECKPOINT" ] && [ -f "$SCIENCEDB_CHECKPOINT" ] && [ ! -f "$SCIENCEDB_OUT" ]; then
        echo "--- 8b: ScienceDB multi-IoU evaluation ---"
        $PYTHON "$SCIENCEDB_EVAL" \
            --checkpoint "$SCIENCEDB_CHECKPOINT" \
            --coco-path "$SCIENCEDB_PATH" \
            --split val \
            --output "$SCIENCEDB_OUT" \
            --device cuda
    elif [ -z "$SCIENCEDB_CHECKPOINT" ]; then
        echo "[ACTION REQUIRED] ScienceDB evaluation was not run. Set SCIENCEDB_CHECKPOINT"
        echo "                  to a checkpoint trained on the five ScienceDB classes."
    elif [ ! -f "$SCIENCEDB_CHECKPOINT" ]; then
        echo "[ERROR] SCIENCEDB_CHECKPOINT does not exist: $SCIENCEDB_CHECKPOINT"
        return 1
    else
        echo "[SKIP] ScienceDB results already exist at $SCIENCEDB_OUT"
    fi

    # 8c — Aggregate all revision results
    echo "--- 8c: Aggregating results ---"
    $PYTHON "$SUMMARIZE" \
        --root "$OUTPUT_ROOT" \
        --split valid \
        --output-prefix "$SUMMARY_DIR/validation"

    # 8d — Generate all LaTeX tables
    echo "--- 8d: Generating paper tables ---"
    local MAKE_CMD=(
        "$PYTHON" "$MAKE_TABLES"
        --registry tools/baseline_registry.yaml
        --summary "$SUMMARY_DIR/validation.summary.csv"
        --out-dir "$PAPER_TABLES"
    )
    if [ -f "$DIAG_DIR/fps_results.json" ]; then
        MAKE_CMD+=(--fps "$DIAG_DIR/fps_results.json")
    fi
    if [ -f "$DEGRADATION_OUT" ]; then
        MAKE_CMD+=(--degradation "$DEGRADATION_OUT")
    fi
    if [ -f "$SCIENCEDB_OUT" ]; then
        MAKE_CMD+=(--sciencedb "$SCIENCEDB_OUT")
    fi
    "${MAKE_CMD[@]}"
}

# ── MAIN ───────────────────────────────────────────────────────────────────

main() {
    banner "Spectral-DETR Revision — Full Experiment Protocol"
    echo "Start: $(date)"
    echo "Output root: $OUTPUT_ROOT"
    echo "Primary seed: $PRIMARY_SEED"
    echo "Extra seeds: ${EXTRA_SEEDS[*]}"
    echo ""

    check_python

    run_core
    run_external
    run_coupling
    run_bands
    run_scu
    run_fps
    run_fig5
    run_diagnostics

    banner "ALL DONE"
    echo "Finished: $(date)"
    echo ""
    echo "Outputs:"
    echo "  LaTeX tables → $PAPER_TABLES/"
    echo "  Value map    → $PAPER_TABLES/value_map.json"
    echo "  Summaries    → $SUMMARY_DIR/"
    echo "  Figures      → $FIGURES_DIR/"
    echo "  Diagnostics  → $DIAG_DIR/"
    echo "  Runs         → $OUTPUT_ROOT/"
    echo ""
    echo "Next steps:"
    echo "  1. Review $PAPER_TABLES/ for generated LaTeX tables."
    echo "  2. Update template.tex with \\input{results/paper_tables/table_xxx}."
    echo "  3. Fill narrative placeholders using $PAPER_TABLES/value_map.json."
}

main
