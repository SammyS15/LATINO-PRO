#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=9:00:00
#SBATCH --account=rrg-lplevass
#SBATCH --gres=gpu:1
#SBATCH --job-name=latino_inpaint
#SBATCH --output=./logs/slurm_%j.out
#SBATCH --error=./logs/slurm_%j.err
#SBATCH --mail-user="sammy.sharief@umontreal.ca"
#SBATCH --mail-type=ALL

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# mkdir -p logs
# mkdir -p .hf_home

module purge
module load python/3.11
module load gcc
module load arrow/23.0.1
source $HOME/links/scratch/LATINO-PRO/latino_env/bin/activate

export PYTHONUNBUFFERED=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME=$HOME/links/scratch/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=$HF_HOME/hub
mkdir -p $HF_HOME/hub/checkpoints
chmod -R u+w $HF_HOME

METHOD="${METHOD:-LATINO}"
IMAGE="${IMAGE:-face}"
PROBLEM="${PROBLEM:-inpainting_squared_mask}"
RUN_TAG="${RUN_TAG:-slurm_${SLURM_JOB_ID:-manual}}"
export IMAGE=cat   # or face
export PROBLEM=inpainting_squared_mask  # or inpainting_uncrop

echo "Repo dir: $REPO_DIR"
echo "Method: $METHOD"
echo "Image config: $IMAGE"
echo "Problem config: $PROBLEM"
echo "Run tag: $RUN_TAG"

cd /home/sammys15/links/scratch/LATINO-PRO

python main_LATINO.py \
    problem="$PROBLEM" \
    image="$IMAGE" \
    log_subfolder="${RUN_TAG}/latino"
