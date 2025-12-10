#!/bin/bash
#$ -P cs523aw      # Specify the SCC project name you want to use
#$ -l h_rt=48:00:00   # Specify the hard time limit for the job
#$ -l mem_per_core=16G
#$ -l gpus=1
#$ -l gpu_c=7.0

source ~/.bashrc
mkdir -p logs

module load miniconda
module load academic-ml/fall-2025
conda activate fall-2025-pyt

python scripts/train.py