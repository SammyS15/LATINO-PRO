#!/bin/bash
# Master orchestrator for LATINO.
#
# Submits one SLURM job per image found in IMAGE_DIR, using the
# generic 'auto' image config and overriding image.path at runtime.
#
# Run from LATENT_METHODS_COMP/LATINO-PRO/:
#   bash master_latino_pro.sh
#
# Override any parameter via environment variable:
#   IMAGE_DIR=/path/to/images PROBLEM=sr_x4 bash master_latino_pro.sh
#
# To test an empty prompt (idea 2):
#   IMAGE_PROMPT="" bash master_latino_pro.sh
#
# Supported PROBLEM values (configs/problem/):
#   inpainting_squared_mask, deblurring_gaussian, deblurring_motion,
#   sr_x4, sr_x8, sr_x16, sr_x32

set -euo pipefail

# ── Parameters (override via env vars) ──────────────────────────────────────
# IMAGE_DIR="${IMAGE_DIR:-/lustre/fsn1/projects/rech/ynx/uxl64xr/latent_model_test_images}"
IMAGE_DIR="${IMAGE_DIR:-/lustre/fsn1/projects/rech/ynx/uxl64xr/Images_Posterior_Method_Test_512}"
PROBLEM="${PROBLEM:-inpainting_squared_mask}"
IMAGE_PROMPT="${IMAGE_PROMPT:-}"
INIT_STRATEGY="${INIT_STRATEGY:-noise}"
RUN_TAG="${RUN_TAG:-master_$(date +%Y%m%d_%H%M%S)}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$SCRIPT_DIR/logs"

echo "============================================================"
echo " LATINO  Multi-Image Run"
echo "  image_dir:    $IMAGE_DIR"
echo "  problem:      $PROBLEM"
echo "  prompt:       '${IMAGE_PROMPT}'"
echo "  init:         $INIT_STRATEGY"
echo "  run_tag:      $RUN_TAG"
echo "============================================================"

# Collect all PNG/JPG images in the directory
mapfile -t IMAGES < <(find "$IMAGE_DIR" -maxdepth 1 -type f \( -iname "*.png" -o -iname "*.jpg" -o -iname "*.jpeg" \) | sort)

if [[ ${#IMAGES[@]} -eq 0 ]]; then
    echo "ERROR: No images found in $IMAGE_DIR"
    exit 1
fi

echo "Found ${#IMAGES[@]} images — submitting one job each."
echo ""

for IMAGE_PATH in "${IMAGES[@]}"; do
    IMAGE_TAG="$(basename "${IMAGE_PATH%.*}")"

    JOB_ID=$(sbatch --parsable \
        --job-name="lpro_${IMAGE_TAG}" \
        --output="$SCRIPT_DIR/logs/${RUN_TAG}_${IMAGE_TAG}_%j.out" \
        --error="$SCRIPT_DIR/logs/${RUN_TAG}_${IMAGE_TAG}_%j.err" \
        --export=ALL,\
IMAGE_PATH="$IMAGE_PATH",\
IMAGE_PROMPT="$IMAGE_PROMPT",\
PROBLEM="$PROBLEM",\
INIT_STRATEGY="$INIT_STRATEGY",\
RUN_TAG="$RUN_TAG" \
        "$SCRIPT_DIR/run_auto.sbatch")

    echo "  Submitted $IMAGE_TAG  →  job $JOB_ID"
done

echo ""
echo "Monitor all:  squeue -u \$USER"
echo "Results:      LATINO-PRO/outputs/${RUN_TAG}/"