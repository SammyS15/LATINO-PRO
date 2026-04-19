#!/bin/bash
#SBATCH --job-name=latino_pro_inpaint
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=2:00:00
#SBATCH --gpus-per-node=1
#SBATCH --account=ynx@v100
#SBATCH -C v100-32g
#SBATCH --output=./logs/apt_pro_slurm_%j.out
#SBATCH --error=./logs/apt_pro_slurm_%j.err
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

PROBLEM="${PROBLEM:-inpainting_squared_mask}"
RUN_TAG="${RUN_TAG:-slurm_${SLURM_JOB_ID:-manual}}"

echo "Work dir:       $(pwd)"
echo "Problem config: $PROBLEM"
echo "Run tag:        $RUN_TAG"

python main_LATINO_PRO.py \
    problem="$PROBLEM" \
    image=apt \
    init_strategy=y_noise \
    log_subfolder="${RUN_TAG}/latino_pro"

