import os
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from datetime import datetime

# Import from updated modules
from datareader import EEGDataset, default_transform, collate_fn
from losses import (
    WeightedCrossEntropyLoss,
    FocalLoss,
    compute_class_weights,
    compute_accuracy,
    compute_per_class_metrics,
    compute_confusion_matrix
)
from models import create_model

# REPRODUCIBILITY
RANDOM_SEED = 2004

def set_seed(seed=RANDOM_SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed()

# TRAINING CONFIGURATION
def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train Mamba for drowsiness detection')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='psg',
                        help='Directory containing EEG data')
    parser.add_argument('--fold', type=int, default=0,
                        help='K-fold index (0-4 for 5-fold CV)')
    parser.add_argument('--n_splits', type=int, default=5,
                        help='Number of folds for cross-validation')
    parser.add_argument('--window_sec', type=int, default=1,
                        help='Window size in seconds')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='mamba', choices=['mamba', 'resnet18'],
                        help='Model architecture')
    parser.add_argument('--d_model', type=int, default=128,
                        help='Hidden dimension for Mamba')
    parser.add_argument('--n_layers', type=int, default=4,
                        help='Number of Mamba layers')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    
    # Loss function
    parser.add_argument('--loss_type', type=str, default='weighted_ce', 
                        choices=['weighted_ce', 'focal'],
                        help='Loss function type')
    parser.add_argument('--focal_gamma', type=float, default=2.0,
                        help='Gamma parameter for focal loss')
    
    # MLflow
    parser.add_argument('--experiment_name', type=str, default='drowsiness_detection',
                        help='MLflow experiment name')
    parser.add_argument('--run_name', type=str, default=None,
                        help='MLflow run name (optional)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for training')
    
    return parser.parse_args()


# TRAINING FUNCTIONS
def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()
    
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    for batch_idx, (signals, targets) in enumerate(pbar):
        # Move to device
        signals = signals.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(signals)  # (B, 3)
        
        # Compute loss
        loss = criterion(logits, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        running_loss += loss.item()
        predictions = torch.argmax(logits, dim=1)  # (B,)
        
        all_predictions.append(predictions.cpu())
        all_targets.append(targets.cpu())
        
        # Update progress bar
        pbar.set_postfix({'loss': loss.item()})
    
    # Compute epoch metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = compute_accuracy(all_predictions, all_targets)
    
    # Per-class metrics
    metrics = compute_per_class_metrics(all_predictions, all_targets, num_classes=3)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1']
    }


def validate(model, val_loader, criterion, device, epoch):
    """Validate the model."""
    model.eval()
    
    running_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
        for signals, targets in pbar:
            # Move to device
            signals = signals.to(device)
            targets = targets.to(device)
            
            # Forward pass
            logits = model(signals)
            loss = criterion(logits, targets)
            
            # Track metrics
            running_loss += loss.item()
            predictions = torch.argmax(logits, dim=1)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            
            pbar.set_postfix({'loss': loss.item()})
    
    # Compute epoch metrics
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = compute_accuracy(all_predictions, all_targets)
    
    # Per-class metrics
    metrics = compute_per_class_metrics(all_predictions, all_targets, num_classes=3)
    
    # Confusion matrix
    cm = compute_confusion_matrix(all_predictions, all_targets, num_classes=3)
    
    return {
        'loss': epoch_loss,
        'accuracy': epoch_acc,
        'precision': metrics['precision'],
        'recall': metrics['recall'],
        'f1': metrics['f1'],
        'confusion_matrix': cm
    }

# MAIN TRAINING LOOP
def main():
    """Main training function."""
    args = parse_args()
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # SETUP DATASETS
    print("\n" + "="*80)
    print("LOADING DATASETS")
    print("="*80)
    
    train_dataset = EEGDataset(
        data_dir=args.data_dir,
        fold=args.fold,
        split='train',
        n_splits=args.n_splits,
        window_sec=args.window_sec,
        transform=default_transform,
        random_offset=True
    )
    
    val_dataset = EEGDataset(
        data_dir=args.data_dir,
        fold=args.fold,
        split='val',
        n_splits=args.n_splits,
        window_sec=args.window_sec,
        transform=default_transform,
        random_offset=False
    )
    
    print(f"Fold {args.fold + 1}/{args.n_splits}")
    print(f"Train windows: {len(train_dataset)}")
    print(f"Val windows: {len(val_dataset)}")
    
    # Compute class weights from training data
    print("\nComputing class weights...")
    all_train_labels = []
    for _, label, *_ in train_dataset:
        if label is not None:
            all_train_labels.append(label)
    
    class_weights = compute_class_weights(np.array(all_train_labels), num_classes=3)
    class_weights = class_weights.to(device)
    print(f"Class weights: {class_weights}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # CREATE MODEL
    print("\n" + "="*80)
    print("CREATING MODEL")
    print("="*80)
    
    model = create_model(
        model_name=args.model,
        in_channels=7,
        num_classes=3,  # 3-class classification
        d_model=args.d_model,
        n_layers=args.n_layers,
        dropout=args.dropout
    )
    model = model.to(device)
    
    print(f"Model: {args.model}")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")
    
    # SETUP TRAINING
    print("\n" + "="*80)
    print("SETUP TRAINING")
    print("="*80)
    
    # Loss function
    if args.loss_type == 'weighted_ce':
        criterion = WeightedCrossEntropyLoss(weight=class_weights)
        print(f"Loss function: Weighted Cross-Entropy")
    elif args.loss_type == 'focal':
        criterion = FocalLoss(alpha=class_weights, gamma=args.focal_gamma)
        print(f"Loss function: Focal Loss (gamma={args.focal_gamma})")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    print(f"Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"Scheduler: CosineAnnealingLR")
    
    # MLFLOW SETUP
    mlflow.set_experiment(args.experiment_name)
    
    run_name = args.run_name or f"{args.model}_fold{args.fold}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_params({
            'model': args.model,
            'fold': args.fold,
            'window_sec': args.window_sec,
            'd_model': args.d_model,
            'n_layers': args.n_layers,
            'dropout': args.dropout,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'loss_type': args.loss_type,
            'epochs': args.epochs,
            'num_params': model.get_num_params(),
            'train_samples': len(train_dataset),
            'val_samples': len(val_dataset)
        })
        
        # TRAINING LOOP
        print("\n" + "="*80)
        print("TRAINING")
        print("="*80)
        
        best_val_acc = 0.0
        best_epoch = 0
        
        for epoch in range(args.epochs):
            # Train
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device, epoch
            )
            
            # Validate
            val_metrics = validate(
                model, val_loader, criterion, device, epoch
            )
            
            # Update learning rate
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            print(f"  LR: {current_lr:.6f}")
            print(f"  Train Loss: {train_metrics['loss']:.4f} | Acc: {train_metrics['accuracy']*100:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f} | Acc: {val_metrics['accuracy']*100:.2f}%")
            
            # Print per-class metrics
            class_names = ['Alert', 'Low Vigilance', 'Drowsy']
            print(f"\n  Validation Per-Class Metrics:")
            for i, name in enumerate(class_names):
                print(f"    {name}:")
                print(f"      Precision: {val_metrics['precision'][i]:.3f}")
                print(f"      Recall:    {val_metrics['recall'][i]:.3f}")
                print(f"      F1:        {val_metrics['f1'][i]:.3f}")
            
            # Log to MLflow
            mlflow.log_metrics({
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy'],
                'learning_rate': current_lr,
                # Per-class metrics
                'val_precision_alert': val_metrics['precision'][0],
                'val_precision_low_vig': val_metrics['precision'][1],
                'val_precision_drowsy': val_metrics['precision'][2],
                'val_recall_alert': val_metrics['recall'][0],
                'val_recall_low_vig': val_metrics['recall'][1],
                'val_recall_drowsy': val_metrics['recall'][2],
                'val_f1_alert': val_metrics['f1'][0],
                'val_f1_low_vig': val_metrics['f1'][1],
                'val_f1_drowsy': val_metrics['f1'][2],
            }, step=epoch)
            
            # Save best model
            if val_metrics['accuracy'] > best_val_acc:
                best_val_acc = val_metrics['accuracy']
                best_epoch = epoch
                
                # Save model checkpoint
                checkpoint_path = f"best_model_fold{args.fold}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_accuracy': best_val_acc,
                    'args': vars(args)
                }, checkpoint_path)
                
                # Log model to MLflow
                mlflow.pytorch.log_model(model, "best_model")
                print(f"  ✅ New best model saved! (Acc: {best_val_acc*100:.2f}%)")
        
        # FINAL RESULTS
        print("\n" + "="*80)
        print("TRAINING COMPLETE")
        print("="*80)
        print(f"Best validation accuracy: {best_val_acc*100:.2f}% (epoch {best_epoch+1})")
        
        # Log best metrics
        mlflow.log_metrics({
            'best_val_accuracy': best_val_acc,
            'best_epoch': best_epoch
        })
        
        # Log confusion matrix as artifact
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Load best model
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get final validation metrics
        final_val_metrics = validate(model, val_loader, criterion, device, epoch=-1)
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            final_val_metrics['confusion_matrix'].numpy(),
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names
        )
        plt.title(f'Confusion Matrix (Fold {args.fold})')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        cm_path = f"confusion_matrix_fold{args.fold}.png"
        plt.savefig(cm_path)
        mlflow.log_artifact(cm_path)
        plt.close()
        
        print(f"\nConfusion matrix saved to: {cm_path}")
        print(f"Model checkpoint saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()