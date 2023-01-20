#!/bin/bash

#SBATCH --job-name=contrastive
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=0-12:00:00  # 12 hours timelimit
#SBATCH --mem=64000MB

source /home/${USER}/.bashrc
conda activate tch

srun python -u /home/gyuseonglee/workspace/biascfr/contrastive.py 
