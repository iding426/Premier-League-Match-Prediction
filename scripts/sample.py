"""
Sample predictions script - generates predictions for recent matches
"""
import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import csv

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Model imports
from models.linear import MLP
from models.transformer import FPLMatchPredictor

# Dataset
from data.dataset import LinearDataset, TransformerDataset

# Arguments
parser = argparse.ArgumentParser(description="Generate sample predictions")
parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "transformer"], help="Which model to use")
parser.add_argument("--checkpoint", type=str, required=True, help="Path to model weights")
parser.add_argument("--data-path", type=str, required=True, help="Path to the dataset CSV file")
parser.add_argument("--output", type=str, default="sample_predictions.csv", help="Output CSV file")
parser.add_argument("--num-samples", type=int, default=50, help="Number of samples to generate")
parser.add_argument("--year", type=int, default=2024, help="Year to sample from")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model
if args.model == "mlp":
    model = MLP().to(device)
elif args.model == "transformer":
    model = FPLMatchPredictor().to(device)

checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.eval()

# Load and filter dataset
df = pd.read_csv(args.data_path)
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'].dt.year == args.year]
    print(f"Found {len(df)} matches from year {args.year}")
else:
    print("Warning: 'date' column not found")

if len(df) == 0:
    print(f"No matches found for year {args.year}")
    sys.exit(1)

# Take last N matches
df = df.tail(args.num_samples)
print(f"Sampling {len(df)} most recent matches from {args.year}")

# Create dataset
if args.model == "mlp":
    ds = LinearDataset(df)
elif args.model == "transformer":
    ds = TransformerDataset(df)

dl = DataLoader(ds, batch_size=4, shuffle=False, drop_last=False)

# Generate predictions
results = []
result_map = {0: 'Home Win', 1: 'Draw', 2: 'Away Win'}

with torch.no_grad():
    for batch_idx, (x, y_result, y_goal_diff) in enumerate(tqdm(dl, desc="Generating Predictions")):
        x = x.to(device)
        
        # Forward pass
        logits, xgd_pred = model(x)
        probs = F.softmax(logits, dim=1)
        predicted_results = torch.argmax(probs, dim=1)
        
        # Store results
        for i in range(len(y_result)):
            idx = batch_idx * 4 + i
            if idx < len(ds):
                match_row = ds.df.iloc[idx]
                results.append({
                    'date': str(match_row.get('date', '')),
                    'home_team': match_row['home_team'],
                    'away_team': match_row['away_team'],
                    'actual_result': result_map[y_result[i].item()],
                    'predicted_result': result_map[predicted_results[i].item()],
                    'home_win_prob': f"{probs[i, 0].item():.1%}",
                    'draw_prob': f"{probs[i, 1].item():.1%}",
                    'away_win_prob': f"{probs[i, 2].item():.1%}",
                    'actual_score': f"{match_row['home_goals']}-{match_row['away_goals']}",
                    'predicted_gd': f"{xgd_pred[i].item():.1f}"
                })

# Save to CSV
with open(args.output, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"\n✅ Saved {len(results)} predictions to {args.output}")
