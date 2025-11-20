# Imports
import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import json
from pathlib import Path


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Model imports
from models.linear import MLP
from models.transformer import FPLMatchPredictor

# Dataset
from data.dataset import LinearDataset, TransformerDataset

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Arguments
parser = argparse.ArgumentParser(description="Evaluate performance of FPL prediction models")
parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "transformer"], help="Which fusion model to evaluate")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to fusion model weights")
parser.add_argument("--output-dir", type=str, default="eval_results", help="Directory to save evaluation results")
parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
parser.add_argument("--dataset", ype=str, required=True, help="Path to eval CSV")

args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

if args.fusion_model == "mlp":
    model = MLP().to(device)
elif args.fusion_model == "transformer":
    model = FPLMatchPredictor().to(device)
else:
    raise ValueError(f"Unknown fusion model: {args.fusion_model}")

# Load Weights
checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    # Checkpoint is wrapped (from training script)
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    # Direct state_dict
    model.load_state_dict(checkpoint)

model.eval() # Set model to eval mode

# Loss Setup
alpha = 1
beta = 1

ce_loss = nn.CrossEntropyLoss()
mse = nn.MSELoss()

# Compute the Overall loss
def loss(
    logits,              # (batch, 3)
    xgd_pred,            # (batch,)
    y_result,            # int labels (0/1/2)
    y_xgd,               # float labels
    alpha=1.0,           # weight for CE
    beta=1.0             # weight for MSE
):
    ce = ce_loss(logits, y_result)
    error = mse(xgd_pred, y_xgd)

    # combined loss
    total_loss = alpha * ce + beta * error
    
    return total_loss, ce, error

# Dataset
if args.model == "mlp":
    ds = LinearDataset()
elif args.model == "transformer":
    ds = TransformerDataset()
else:
    raise ValueError(f"Unknown fusion model: {args.fusion_model}")

dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

