#!/bin/bash
# set -euo pipefail
# IFS=$'\n\t'

# #SBATCH --job-name=nlu-experiment
# #SBATCH --nodes=1
# #SBATCH --mem-per-cpu=32GB
# #SBATCH --cpus-per-task=2
# #SBATCH --time=96:00:00
# #SBATCH --gres=gpu:1

module purge
module load cuda/10.2.89
module load anaconda3/2020.07

conda activate nlu-final-project

./experiment-script-jigsaw.py $@
