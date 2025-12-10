"""
Training script for Football Match Prediction
Trains both Transformer and MLP models with temporal split
"""

import sys
import pathlib

# Add project root to sys.path
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from datetime import datetime

# Import your modules
from data.dataset import TransformerDataset, LinearDataset
from models.linear import MLP


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        device,
        model_name,
        save_dir="models_2",
        lr=1e-4,
        weight_decay=1e-3,  # Increased from 1e-5 for stronger regularization
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.model_name = model_name
        self.save_dir = save_dir
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)  # Prevents overconfidence
        self.mse_loss = nn.MSELoss()
        
        # Tracking
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_ce_loss = 0
        total_mse_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f"Training {self.model_name}")
        
        for batch in pbar:
            x, y_result, y_goal_diff = batch
            x = x.to(self.device)
            y_result = y_result.to(self.device)
            y_goal_diff = y_goal_diff.to(self.device)
            
            # Forward
            probs, goal_diff_pred = self.model(x)
            
            # Classification loss
            ce_loss = self.ce_loss(probs, y_result)
            
            # Regression loss
            mse_loss = self.mse_loss(goal_diff_pred, y_goal_diff)
            
            # Combined loss (equal weighting)
            loss = ce_loss + mse_loss
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_mse_loss += mse_loss.item()
            
            _, predicted = probs.max(1)
            total += y_result.size(0)
            correct += predicted.eq(y_result).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_ce = total_ce_loss / len(self.train_loader)
        avg_mse = total_mse_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, avg_ce, avg_mse, accuracy
    
    @torch.no_grad()
    def evaluate(self, loader, split_name="Val"):
        """Evaluate on validation or test set"""
        self.model.eval()
        total_loss = 0
        total_ce_loss = 0
        total_mse_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(loader, desc=f"Evaluating {split_name}")
        
        for batch in pbar:
            x, y_result, y_goal_diff = batch
            x = x.to(self.device)
            y_result = y_result.to(self.device)
            y_goal_diff = y_goal_diff.to(self.device)
            
            # Forward
            probs, goal_diff_pred = self.model(x)
            
            # Losses
            ce_loss = self.ce_loss(probs, y_result)
            mse_loss = self.mse_loss(goal_diff_pred, y_goal_diff)
            loss = ce_loss + mse_loss
            
            # Track metrics
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            total_mse_loss += mse_loss.item()
            
            _, predicted = probs.max(1)
            total += y_result.size(0)
            correct += predicted.eq(y_result).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        avg_loss = total_loss / len(loader)
        avg_ce = total_ce_loss / len(loader)
        avg_mse = total_mse_loss / len(loader)
        accuracy = 100. * correct / total
        
        return avg_loss, avg_ce, avg_mse, accuracy
    
    def train(self, epochs):
        """Full training loop"""
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_ce, train_mse, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_ce, val_mse, val_acc = self.evaluate(
                self.val_loader, split_name="Val"
            )
            
            # Log
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            
            print(f"\nTrain Loss: {train_loss:.4f} (CE: {train_ce:.4f}, MSE: {train_mse:.4f}) | Acc: {train_acc:.2f}%")
            print(f"Val   Loss: {val_loss:.4f} (CE: {val_ce:.4f}, MSE: {val_mse:.4f}) | Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss, val_acc, best=True)
                print(f"âœ… New best model! Val loss: {val_loss:.4f}")
            
            # Save periodic checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(epoch, val_loss, val_acc, best=False)
        
        # Final evaluation on test set
        print(f"\n{'='*60}")
        print(f"Final Test Evaluation for {self.model_name}")
        print(f"{'='*60}\n")
        
        # Load best model
        self.load_checkpoint(best=True)
        test_loss, test_ce, test_mse, test_acc = self.evaluate(
            self.test_loader, split_name="Test"
        )
        
        print(f"\nTest Loss: {test_loss:.4f} (CE: {test_ce:.4f}, MSE: {test_mse:.4f}) | Acc: {test_acc:.2f}%")
        
        return {
            'test_loss': test_loss,
            'test_ce': test_ce,
            'test_mse': test_mse,
            'test_acc': test_acc,
        }
    
    def save_checkpoint(self, epoch, val_loss, val_acc, best=False):
        """Save model checkpoint"""
        suffix = "best" if best else f"epoch_{epoch+1}"
        path = os.path.join(self.save_dir, f"{self.model_name}_{suffix}.pt")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
        }, path)
    
    def load_checkpoint(self, best=True):
        """Load model checkpoint"""
        suffix = "best" if best else "latest"
        path = os.path.join(self.save_dir, f"{self.model_name}_{suffix}.pt")
        
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {path}")


def temporal_split(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    Split dataset temporally (by date order).
    
    Returns indices for train, val, test splits.
    """
    total_size = len(dataset)
    
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    train_indices = list(range(0, train_size))
    val_indices = list(range(train_size, train_size + val_size))
    test_indices = list(range(train_size + val_size, total_size))
    
    print(f"Temporal Split:")
    print(f"  Train: {len(train_indices)} samples ({train_ratio*100:.0f}%)")
    print(f"  Val:   {len(val_indices)} samples ({val_ratio*100:.0f}%)")
    print(f"  Test:  {len(test_indices)} samples ({test_ratio*100:.0f}%)")
    
    return train_indices, val_indices, test_indices


def main():
    # Configuration
    DATA_PATH = "data/understat_match_1524.csv"
    WEATHER_CACHE_PATH = "data/weather_cache.csv"
    BATCH_SIZE = 64
    EPOCHS = 100  # Increased from 50 to allow more training with regularization
    LR = 1e-4
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\n{'='*60}")
    print(f"Football Match Prediction - Training")
    print(f"{'='*60}\n")
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LR}")
    
    # Load data
    print(f"\nLoading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} matches")
    
    # Create datasets
    print("\nCreating datasets...")
    
    # # Transformer dataset
    # transformer_dataset = TransformerDataset(
    #     df,
    #     team_history_len=20,
    #     team_feature_dim=32,
    #     match_feature_dim=8,
    #     fetch_weather=True,
    #     weather_cache_path=WEATHER_CACHE_PATH
    # )
    
    # Linear dataset
    linear_dataset = LinearDataset(
        df,
        team_history_len=20,
        team_feature_dim=16,  # Increased from 12 to accommodate 14 features + buffer
        match_feature_dim=8,
        fetch_weather=True,
        weather_cache_path=WEATHER_CACHE_PATH
    )
    
    # Temporal split
    train_idx, val_idx, test_idx = temporal_split(linear_dataset)
    
    # # Create dataloaders - Transformer
    # train_loader_transformer = DataLoader(
    #     Subset(transformer_dataset, train_idx),
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=4,
    #     pin_memory=True
    # )
    # val_loader_transformer = DataLoader(
    #     Subset(transformer_dataset, val_idx),
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True
    # )
    # test_loader_transformer = DataLoader(
    #     Subset(transformer_dataset, test_idx),
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True
    # )
    
    # Create dataloaders - MLP
    train_loader_mlp = DataLoader(
        Subset(linear_dataset, train_idx),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Single process only
        pin_memory=True
    )
    val_loader_mlp = DataLoader(
        Subset(linear_dataset, val_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Single process only
        pin_memory=True
    )
    test_loader_mlp = DataLoader(
        Subset(linear_dataset, test_idx),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,  # Single process only
        pin_memory=True
    )
    
    # # ========== Train Transformer ==========
    # print("\n" + "="*60)
    # print("TRAINING TRANSFORMER MODEL")
    # print("="*60)
    
    # transformer_model = FPLMatchPredictor(
    #     input_dim_team=32,
    #     input_dim_match=8,
    #     model_dim=128,
    #     num_heads=4,
    #     depth_team=2,
    #     depth_global=3,
    #     dropout=0.1,
    # )
    
    # transformer_trainer = Trainer(
    #     model=transformer_model,
    #     train_loader=train_loader_transformer,
    #     val_loader=val_loader_transformer,
    #     test_loader=test_loader_transformer,
    #     device=DEVICE,
    #     model_name="transformer",
    #     lr=LR,
    # )
    
    # transformer_results = transformer_trainer.train(epochs=EPOCHS)
    
    print("TRAINING MLP MODEL")
    
    # Input: (20 matches * 16 features * 2 teams) + 11 match features = 651
    mlp_model = MLP(input_space=651)
    
    mlp_trainer = Trainer(
        model=mlp_model,
        train_loader=train_loader_mlp,
        val_loader=val_loader_mlp,
        test_loader=test_loader_mlp,
        device=DEVICE,
        model_name="mlp",
        lr=5e-5,  # Reduced from 1e-4 for more stable training
        weight_decay=1e-3,  # Stronger L2 regularization
    )
    
    mlp_results = mlp_trainer.train(epochs=EPOCHS)
    
    print("FINAL RESULTS SUMMARY")
    
    # print(f"\nTransformer:")
    # print(f"  Test Loss: {transformer_results['test_loss']:.4f}")
    # print(f"  Test CE:   {transformer_results['test_ce']:.4f}")
    # print(f"  Test MSE:  {transformer_results['test_mse']:.4f}")
    # print(f"  Test Acc:  {transformer_results['test_acc']:.2f}%")
    
    print(f"\nMLP:")
    print(f"  Test Loss: {mlp_results['test_loss']:.4f}")
    print(f"  Test CE:   {mlp_results['test_ce']:.4f}")
    print(f"  Test MSE:  {mlp_results['test_mse']:.4f}")
    print(f"  Test Acc:  {mlp_results['test_acc']:.2f}%")
    
    print(f"Training complete! Models saved in models")
    # print(f"   - models/transformer_best.pt")
    print(f"   - models/mlp_best.pt")


if __name__ == "__main__":
    main()