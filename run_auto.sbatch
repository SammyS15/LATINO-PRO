#!/bin/bash
#SBATCH --job-name=latino_array
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=0:40:00
#SBATCH --gpus-per-node=1
#SBATCH --account=ynx@v100
#SBATCH -C v100-32g
#SBATCH --output=./logs/latino_array_slurm_%j.out
#SBATCH --error=./logs/latino_array_slurm_%j.err
#SBATCH --mail-user="sammy.sharief@umontreal.ca"
#SBATCH --mail-type=ALL

set -euo pipefail

cd /lustre/fsn1/projects/rech/ynx/uxl64xr/LATINO-PRO/
module purge
module load python/3.11.5
module load gcc
source .venv/bin/activate

export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

unset HUGGINGFACE_HUB_CACHE
export HF_HOME=$WORK/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$HF_HOME/hub
export HYDRA_FULL_ERROR=1

# These are injected by master_LATINO.sh via --export
METHOD="${METHOD:-LATINO-1.5}"
PROBLEM="${PROBLEM:-inpainting_squared_mask}"
IMAGE_PATH="${IMAGE_PATH:?IMAGE_PATH must be set}"
IMAGE_PROMPT="${IMAGE_PROMPT:-a high-quality photograph}"
INIT_STRATEGY="${INIT_STRATEGY:-noise}"
RUN_TAG="${RUN_TAG:-slurm_${SLURM_JOB_ID}}"
SD15_MODEL_PATH="${SD15_MODEL_PATH:-/lustre/fswork/projects/rech/ynx/uxl64xr/models/sd15}"
LCM_LORA_PATH="${LCM_LORA_PATH:-/lustre/fswork/projects/rech/ynx/uxl64xr/models/lcm-lora-sdv1-5/pytorch_lora_weights.safetensors}"
MASK_SIZE="${MASK_SIZE:-512}"
MASK_REFERENCE_RESOLUTION="${MASK_REFERENCE_RESOLUTION:-1024}"



IMAGE_TAG="$(basename "${IMAGE_PATH%.*}")"

export SD15_MODEL_PATH
export LCM_LORA_PATH

echo "Job ID:         $SLURM_JOB_ID"
echo "Method:         $METHOD"
echo "Problem config: $PROBLEM"
echo "Image:          $IMAGE_PATH  ($IMAGE_TAG)"
echo "Prompt:         '${IMAGE_PROMPT}'"
echo "Problem:        $PROBLEM"
echo "Run tag:        $RUN_TAG"
echo "SD15 path:      $SD15_MODEL_PATH"
echo "LCM LoRA path:  $LCM_LORA_PATH"
echo "Mask size:      $MASK_SIZE"
echo "Mask ref res:   $MASK_REFERENCE_RESOLUTION"

python main_LATINO.py \
    model="$METHOD" \
    problem="$PROBLEM" \
    problem.mask_size="$MASK_SIZE" \
    problem.mask_reference_resolution="$MASK_REFERENCE_RESOLUTION" \
    image=auto \
    "image.path=${IMAGE_PATH}" \
    "image.prompt='${IMAGE_PROMPT}'" \
    init_strategy="$INIT_STRATEGY" \
    log_subfolder="${RUN_TAG}/${IMAGE_TAG}"