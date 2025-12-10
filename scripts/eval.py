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
import pandas as pd


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
parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "transformer"], help="Which model to evaluate")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights")
parser.add_argument("--output-dir", type=str, default="eval_results", help="Directory to save evaluation results")
parser.add_argument("--batch-size", type=int, default=4, help="Batch size for evaluation")
parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset CSV file")

args = parser.parse_args()

output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

if args.model == "mlp":
    model = MLP().to(device)
elif args.model == "transformer":
    model = FPLMatchPredictor().to(device)
else:
    raise ValueError(f"Unknown model: {args.model}")

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

# Load and filter dataset for years 2022-2024
df = pd.read_csv(args.data_path)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year.isin([2022, 2023, 2024])]
    print(f"Filtered to {len(df)} matches from years 2022-2024")
else:
    print("Warning: 'date' column not found, using all data")

# Dataset
if args.model == "mlp":
    ds = LinearDataset(df)
elif args.model == "transformer":
    ds = TransformerDataset(df)
else:
    raise ValueError(f"Unknown model: {args.model}")

dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

# Eval on years 2022-2024 only
all_losses = []
all_predictions = []
all_labels = []
all_gd_preds = []
all_gd_true = []
sample_results = []  # Store detailed results for a sample of matches

with torch.no_grad():
    for batch_idx, (x, y_result, y_goal_diff) in enumerate(tqdm(dl, desc="Evaluating Model")):
        # Move to device
        x = x.to(device)
        y_result = y_result.to(device)
        y_goal_diff = y_goal_diff.to(device)

        # Forward pass
        logits, xgd_pred = model(x)

        # Compute loss
        batch_loss, ce, error = loss(
            logits,
            xgd_pred,
            y_result,
            y_goal_diff,
            alpha=alpha,
            beta=beta
        )

        all_losses.append({
            'total_loss': batch_loss.item(),
            'ce_loss': ce.item(),
            'mse_loss': error.item()
        })
        
        # Store predictions and labels for accuracy calculation
        probs = F.softmax(logits, dim=1)
        predicted_results = torch.argmax(probs, dim=1)
        all_predictions.extend(predicted_results.cpu().numpy())
        all_labels.extend(y_result.cpu().numpy())
        all_gd_preds.extend(xgd_pred.cpu().numpy())
        all_gd_true.extend(y_goal_diff.cpu().numpy())
        
        # Save sample results (first 50 matches)
        if len(sample_results) < 50:
            for i in range(len(y_result)):
                if len(sample_results) >= 50:
                    break
                idx = batch_idx * args.batch_size + i
                if idx < len(ds):
                    match_row = ds.df.iloc[idx]
                    result_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}
                    sample_results.append({
                        'date': str(match_row.get('date', '')),
                        'home_team': match_row['home_team'],
                        'away_team': match_row['away_team'],
                        'actual_result': result_map[y_result[i].item()],
                        'predicted_result': result_map[predicted_results.cpu().numpy()[batch_idx * args.batch_size + i]],
                        'home_win_prob': f"{probs[i, 0].item():.2%}",
                        'draw_prob': f"{probs[i, 1].item():.2%}",
                        'away_win_prob': f"{probs[i, 2].item():.2%}",
                        'actual_gd': f"{y_goal_diff[i].item():.1f}",
                        'predicted_gd': f"{xgd_pred[i].item():.1f}"
                    })

# Aggregate results
total_loss = np.mean([l['total_loss'] for l in all_losses])
ce_loss_avg = np.mean([l['ce_loss'] for l in all_losses])
mse_loss_avg = np.mean([l['mse_loss'] for l in all_losses])

# Calculate accuracy metrics
all_predictions = np.array(all_predictions)
all_labels = np.array(all_labels)
all_gd_preds = np.array(all_gd_preds)
all_gd_true = np.array(all_gd_true)

# Overall accuracy
overall_accuracy = np.mean(all_predictions == all_labels)

# Per-class accuracy
home_win_mask = all_labels == 0
draw_mask = all_labels == 1
away_win_mask = all_labels == 2

home_win_acc = np.mean(all_predictions[home_win_mask] == 0) if home_win_mask.sum() > 0 else 0
draw_acc = np.mean(all_predictions[draw_mask] == 1) if draw_mask.sum() > 0 else 0
away_win_acc = np.mean(all_predictions[away_win_mask] == 2) if away_win_mask.sum() > 0 else 0

# Average goal difference error (MAE)
avg_gd_error = np.mean(np.abs(all_gd_preds - all_gd_true))

results = {
    'total_loss': total_loss,
    'ce_loss': ce_loss_avg,
    'mse_loss': mse_loss_avg,
    'overall_accuracy': float(overall_accuracy),
    'home_win_accuracy': float(home_win_acc),
    'draw_accuracy': float(draw_acc),
    'away_win_accuracy': float(away_win_acc),
    'avg_goal_diff_error': float(avg_gd_error),
    'sample_predictions': sample_results
}

# Save results
results_path = output_dir / (args.model + "_results.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)

# Also save sample as CSV for easier viewing
if sample_results:
    import csv
    csv_path = output_dir / (args.model + "_sample_predictions.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sample_results[0].keys())
        writer.writeheader()
        writer.writerows(sample_results)
    print(f"Sample predictions saved to {csv_path}")

print(f"Evaluation results saved to {results_path}")
print(f"\nAccuracy Metrics:")
print(f"  Overall: {overall_accuracy:.2%}")
print(f"  Home Win: {home_win_acc:.2%}")
print(f"  Draw: {draw_acc:.2%}")
print(f"  Away Win: {away_win_acc:.2%}")
print(f"  Avg GD Error: {avg_gd_error:.3f}")