#!/bin/bash
#SBATCH --job-name=latino_inpaint_metrics
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=9:00:00
#SBATCH --gpus-per-node=1
#SBATCH --account=ynx@v100
#SBATCH -C v100-32g
#SBATCH --output=./logs/slurm_metrics_%j.out
#SBATCH --error=./logs/slurm_metrics_%j.err
#SBATCH --mail-user="sammy.sharief@umontreal.ca"
#SBATCH --mail-type=ALL

set -euo pipefail

cd /lustre/fsn1/projects/rech/ynx/uxl64xr/LATINO-PRO/
module purge
module load python/3.11.5       # adjust to your preferred version
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

# --- Configuration (override via: METHOD=foo sbatch script.sh) ---
METHOD="${METHOD:-LATINO}"
IMAGE="${IMAGE:-cat}"
PROBLEM="${PROBLEM:-inpainting_squared_mask}"
RUN_TAG="${RUN_TAG:-slurm_${SLURM_JOB_ID:-manual}}"

echo "Work dir:       $(pwd)"
echo "Method:         $METHOD"
echo "Image config:   $IMAGE"
echo "Problem config: $PROBLEM"
echo "Run tag:        $RUN_TAG"

python main_LATINO_Metrics.py \
    problem="$PROBLEM" \
    image="$IMAGE" \
    log_subfolder="${RUN_TAG}/latino"
