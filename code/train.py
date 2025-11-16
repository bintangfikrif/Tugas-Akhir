import os
import argparse
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import mlflow
import mlflow.pytorch
from torchinfo import summary

from datareader import EEGDataset, default_transform, collate_fn
from models import create_model
from losses import OrdinalRegressionLoss, logits_to_class, compute_mae, compute_accuracy_with_tolerance
from config import Config


class EarlyStopping:
    
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def plot_training_history(history, save_path):
    """Plot and save training history."""
    epochs = len(history['train_loss'])
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0, 0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # MAE
    axes[0, 1].plot(history['train_mae'], label='Train MAE', marker='o')
    axes[0, 1].plot(history['val_mae'], label='Val MAE', marker='o')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('MAE')
    axes[0, 1].set_title('Mean Absolute Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Accuracy (exact)
    axes[0, 2].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[0, 2].plot(history['val_acc'], label='Val Acc', marker='o')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Accuracy (%)')
    axes[0, 2].set_title('Accuracy (Exact Match)')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Accuracy (±1)
    axes[1, 0].plot(history['train_acc_tol1'], label='Train Acc ±1', marker='o')
    axes[1, 0].plot(history['val_acc_tol1'], label='Val Acc ±1', marker='o')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy (%)')
    axes[1, 0].set_title('Accuracy (±1 Tolerance)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if 'lr' in history:
        axes[1, 1].plot(history['lr'], marker='o', color='green')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    # Confusion info (val set, last epoch)
    if 'val_pred_dist' in history and len(history['val_pred_dist']) > 0:
        pred_dist = history['val_pred_dist'][-1]
        true_dist = history['val_true_dist'][-1]
        
        x = np.arange(9)
        width = 0.35
        axes[1, 2].bar(x - width/2, true_dist, width, label='True', alpha=0.8)
        axes[1, 2].bar(x + width/2, pred_dist, width, label='Predicted', alpha=0.8)
        axes[1, 2].set_xlabel('KSS Level')
        axes[1, 2].set_ylabel('Count')
        axes[1, 2].set_title('Class Distribution (Last Epoch)')
        axes[1, 2].set_xticks(x)
        axes[1, 2].set_xticklabels([f'{i+1}' for i in range(9)])
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training history plot saved to: {save_path}")
    plt.close()


def train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch, args):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
    
    for batch_idx, (signals, labels, ordinal_labels) in enumerate(pbar):
        signals = signals.to(device)
        labels = labels.to(device)
        ordinal_labels = ordinal_labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with autocast(enabled=args.use_amp):
            ordinal_logits = model(signals)
            loss = criterion(ordinal_logits, ordinal_labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Get predictions
        with torch.no_grad():
            predictions = logits_to_class(ordinal_logits)
            all_predictions.append(predictions.cpu())
            all_targets.append(labels.cpu())
        
        running_loss += loss.item() * signals.size(0)
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_mae = compute_mae(all_predictions, all_targets).item()
    epoch_acc = compute_accuracy_with_tolerance(all_predictions, all_targets, tolerance=0).item() * 100
    epoch_acc_tol1 = compute_accuracy_with_tolerance(all_predictions, all_targets, tolerance=1).item() * 100
    
    return epoch_loss, epoch_mae, epoch_acc, epoch_acc_tol1


def validate(model, val_loader, criterion, device, epoch, args):
    """Validate the model."""
    model.eval()
    
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
    
    with torch.no_grad():
        for signals, labels, ordinal_labels in pbar:
            signals = signals.to(device)
            labels = labels.to(device)
            ordinal_labels = ordinal_labels.to(device)
            
            # Forward pass
            with autocast(enabled=args.use_amp):
                ordinal_logits = model(signals)
                loss = criterion(ordinal_logits, ordinal_labels)
            
            # Get predictions
            predictions = logits_to_class(ordinal_logits)
            all_predictions.append(predictions.cpu())
            all_targets.append(labels.cpu())
            
            running_loss += loss.item() * signals.size(0)
            
            # Update progress bar
            pbar.set_postfix({'loss': loss.item()})
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_mae = compute_mae(all_predictions, all_targets).item()
    epoch_acc = compute_accuracy_with_tolerance(all_predictions, all_targets, tolerance=0).item() * 100
    epoch_acc_tol1 = compute_accuracy_with_tolerance(all_predictions, all_targets, tolerance=1).item() * 100
    
    # Class distributions
    pred_dist = torch.bincount(all_predictions, minlength=9).cpu().numpy()
    true_dist = torch.bincount(all_targets, minlength=9).cpu().numpy()
    
    return epoch_loss, epoch_mae, epoch_acc, epoch_acc_tol1, pred_dist, true_dist


def train(args):
    """Main training function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"\n{'='*80}")
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
    print(f"{'='*80}\n")
    
    # Setup MLflow
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment(args.mlflow_experiment_name)
    
    with mlflow.start_run(run_name=f"{args.model_name}_fold{args.fold}"):
        
        # Log parameters
        mlflow.log_params({
            'model_name': args.model_name,
            'fold': args.fold,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'window_sec': args.window_sec,
            'num_classes': args.num_classes,
            'd_model': args.d_model,
            'n_layers': args.n_layers,
            'd_state': args.d_state,
            'd_conv': args.d_conv,
            'expand': args.expand,
            'use_amp': args.use_amp,
            'device': str(device),
        })
        
        # Create datasets
        print("Loading datasets...")
        train_dataset = EEGDataset(
            data_dir=args.data_dir,
            split='train',
            fold=args.fold,
            n_splits=args.n_splits,
            window_sec=args.window_sec,
            transform=default_transform,
            random_offset=True
        )
        
        val_dataset = EEGDataset(
            data_dir=args.data_dir,
            split='val',
            fold=args.fold,
            n_splits=args.n_splits,
            window_sec=args.window_sec,
            transform=default_transform,
            random_offset=False
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=args.num_workers,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        print(f"\nDataset loaded:")
        print(f"  Train: {len(train_dataset)} windows")
        print(f"  Val: {len(val_dataset)} windows")
        
        # Create model
        print(f"\nCreating {args.model_name} model...")
        model = create_model(
            model_name=args.model_name,
            in_channels=len(train_dataset.TARGET_CHANNELS),
            num_classes=args.num_classes,
            d_model=args.d_model,
            n_layers=args.n_layers,
            d_state=args.d_state,
            d_conv=args.d_conv,
            expand=args.expand,
            dropout=args.dropout
        )
        model.to(device)
        
        # Print model architecture
        print(f"\n{'='*80}")
        print("MODEL ARCHITECTURE")
        print(f"{'='*80}")
        
        sample_input = torch.randn(
            args.batch_size,
            len(train_dataset.TARGET_CHANNELS),
            train_dataset.sample_len
        ).to(device)
        
        summary(model, input_data=sample_input, depth=3, 
                col_names=["input_size", "output_size", "num_params", "trainable"],
                row_settings=["var_names"])
        
        print(f"\n{'='*80}")
        print(f"Total parameters: {model.get_num_params():,}")
        print(f"Trainable parameters: {model.get_num_trainable_params():,}")
        print(f"{'='*80}\n")
        
        # Log model info
        mlflow.log_param('total_params', model.get_num_params())
        mlflow.log_param('trainable_params', model.get_num_trainable_params())
        
        # Loss function
        criterion = OrdinalRegressionLoss(num_classes=args.num_classes)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.scheduler_factor,
            patience=args.scheduler_patience,
            verbose=True
        )
        
        # Mixed precision scaler
        scaler = GradScaler(enabled=args.use_amp)
        
        # Early stopping
        early_stopping = EarlyStopping(patience=args.early_stopping_patience, verbose=True)
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_mae': [], 'val_mae': [],
            'train_acc': [], 'val_acc': [],
            'train_acc_tol1': [], 'val_acc_tol1': [],
            'lr': [],
            'val_pred_dist': [], 'val_true_dist': []
        }
        
        best_val_loss = float('inf')
        
        # Training loop
        print(f"\n{'='*80}")
        print("TRAINING START")
        print(f"{'='*80}\n")
        
        start_time = time.time()
        
        for epoch in range(args.epochs):
            
            # Train
            train_loss, train_mae, train_acc, train_acc_tol1 = train_one_epoch(
                model, train_loader, criterion, optimizer, device, scaler, epoch, args
            )
            
            # Validate
            val_loss, val_mae, val_acc, val_acc_tol1, pred_dist, true_dist = validate(
                model, val_loader, criterion, device, epoch, args
            )
            
            # Update learning rate
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            # Save history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_mae'].append(train_mae)
            history['val_mae'].append(val_mae)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            history['train_acc_tol1'].append(train_acc_tol1)
            history['val_acc_tol1'].append(val_acc_tol1)
            history['lr'].append(current_lr)
            history['val_pred_dist'].append(pred_dist)
            history['val_true_dist'].append(true_dist)
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'train_acc_tol1': train_acc_tol1,
                'val_acc_tol1': val_acc_tol1,
                'learning_rate': current_lr
            }, step=epoch)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{args.epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"  Train MAE: {train_mae:.4f} | Val MAE: {val_mae:.4f}")
            print(f"  Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%")
            print(f"  Train Acc±1: {train_acc_tol1:.2f}% | Val Acc±1: {val_acc_tol1:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            
            # Save checkpoint
            os.makedirs(args.checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(args.checkpoint_dir, f'model_epoch{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'history': history
            }, checkpoint_path)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
                torch.save(model.state_dict(), best_model_path)
                print(f"  ✓ Best model saved! (Val Loss: {val_loss:.4f})")
                
                # Log best model to MLflow
                mlflow.pytorch.log_model(model, "best_model")
            
            print(f"{'='*80}\n")
            
            # Early stopping
            early_stopping(val_loss)
            if early_stopping.early_stop:
                print("Early stopping triggered!")
                break
        
        # Training finished
        elapsed_time = time.time() - start_time
        print(f"\n{'='*80}")
        print("TRAINING FINISHED")
        print(f"{'='*80}")
        print(f"Total time: {elapsed_time/60:.2f} minutes")
        print(f"Best validation loss: {best_val_loss:.4f}")
        
        # Plot training history
        plot_path = os.path.join(args.checkpoint_dir, 'training_history.png')
        plot_training_history(history, plot_path)
        
        # Log plot to MLflow
        mlflow.log_artifact(plot_path)
        
        # Log best metrics
        best_epoch = np.argmin(history['val_loss'])
        mlflow.log_metrics({
            'best_epoch': best_epoch,
            'best_val_loss': history['val_loss'][best_epoch],
            'best_val_mae': history['val_mae'][best_epoch],
            'best_val_acc': history['val_acc'][best_epoch],
            'best_val_acc_tol1': history['val_acc_tol1'][best_epoch],
        })
        
        print(f"\nMLflow run completed. Run ID: {mlflow.active_run().info.run_id}")
        print(f"View results: mlflow ui --backend-store-uri {args.mlflow_tracking_uri}")


def parse_args():
    parser = argparse.ArgumentParser(description='Train Drowsiness Detection Model')
    
    # Data
    parser.add_argument('--data-dir', type=str, default='psg', help='Directory with EDF files')
    parser.add_argument('--window-sec', type=int, default=5, help='Window size in seconds')
    
    # Model
    parser.add_argument('--model-name', type=str, default='mamba', choices=['mamba', 'resnet18'],
                       help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=9, help='Number of KSS classes')
    parser.add_argument('--d-model', type=int, default=128, help='Hidden dimension for Mamba')
    parser.add_argument('--n-layers', type=int, default=4, help='Number of Mamba layers')
    parser.add_argument('--d-state', type=int, default=16, help='SSM state expansion factor')
    parser.add_argument('--d-conv', type=int, default=4, help='Local convolution width')
    parser.add_argument('--expand', type=int, default=2, help='Block expansion factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight decay')
    
    # Scheduler
    parser.add_argument('--scheduler-patience', type=int, default=5, help='LR scheduler patience')
    parser.add_argument('--scheduler-factor', type=float, default=0.5, help='LR reduction factor')
    
    # Early stopping
    parser.add_argument('--early-stopping-patience', type=int, default=10, help='Early stopping patience')
    
    # Cross-validation
    parser.add_argument('--fold', type=int, default=0, help='Fold number for cross-validation')
    parser.add_argument('--n-splits', type=int, default=5, help='Number of folds')
    
    # System
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader num_workers')
    parser.add_argument('--use-cuda', action='store_true', default=True, help='Use CUDA if available')
    parser.add_argument('--use-amp', action='store_true', default=True, help='Use automatic mixed precision')
    
    # Paths
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--mlflow-tracking-uri', type=str, default='./mlruns', help='MLflow tracking URI')
    parser.add_argument('--mlflow-experiment-name', type=str, default='drowsiness-detection-mamba',
                       help='MLflow experiment name')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(args)
