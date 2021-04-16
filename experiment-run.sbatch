#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

#SBATCH --job-name=nlu-experiment
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=32GB
#SBATCH --cpus-per-task=2
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1

module purge
module load cuda/11.1.74
module load anaconda3/2020.07
source activate transformers-dev

./experiment-script.py $@