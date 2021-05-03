#!/bin/bash

set -euox pipefail

#SBATCH --job-name=nlu-experiment
#SBATCH --nodes=1
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=nlu_experiment%A_%a.out

module purge
module load cuda/10.2.89
module load anaconda3/2020.07

echo "Loading activate"
set +u
source activate nlu-final-project
set -u


python $(realpath ./experiment_script.py) $@