#!/bin/bash
#SBATCH --mail-user=nir.yarden@mail.huji.ac.il
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:a5000,vmem:40g
#SBATCH --mem-per-cpu=40g
#SBATCH --time=2-0
#SBATCH -c4
#SBATCH --output=/cs/labs/roys/nir.yarden/other/solar_rl/run_outputs/base_run/base_run.log
#SBATCH --job-name=base_run

RUN_NAME="base_run"

export PATH="$HOME/.local/bin:$PATH"

module load cuda
module load nvidia
nvidia-smi
cd /cs/labs/roys/nir.yarden/other/solar_rl
source /cs/labs/roys/nir.yarden/thesis/synthetic/mamba-contextual-selection/.venv/bin/activate

python3 /cs/labs/roys/nir.yarden/other/solar_rl/main.py \
  --hyperparameters "${RUN_NAME}" \
  --train \
  --wandb
