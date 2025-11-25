#!/bin/bash
#$ -P cs599dg      # Specify the SCC project name you want to use
#$ -l h_rt=48:00:00   # Specify the hard time limit for the job
#$ -l mem_per_core=16G
#$ -l gpus=1

source ~/.bashrc
mkdir -p logs

module load miniconda
module load academic-ml/fall-2025
conda activate fall-2025-pyt

# MLP Eval
python scripts/eval.py --model mlp --checkpoint weights/mlp_best.pt --data-path data/understat_match_1524.csv

# Transformer Eval
python scripts/eval.py --model transformer --checkpoint weights/transformer_best.pt --data-path data/understat_match_1524.csv