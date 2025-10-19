import os
import argparse
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 

from datareader import EEGDataset
from models import resnet18_1d 
from torchinfo import summary

def default_transform(x: torch.Tensor) -> torch.Tensor:
    """Per-channel normalization: subtract mean and divide by std.
    x shape: (channels, samples)
    """
    # compute per-channel mean/std
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    return (x - mean) / (std + 1e-6)


def collate_fn(batch):
    # batch is list of tuples: (signals_tensor, label, ...)
    signals = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    signals = torch.stack(signals, dim=0)
    labels = torch.tensor([int(l) if l is not None else 0 for l in labels], dtype=torch.long)
    return signals, labels

def plot_history(checkpoint_dir, epochs, train_loss, val_loss, val_acc):
    """Plots training history and saves it to a file."""
    epoch_range = range(1, epochs + 1)
    
    plt.figure(figsize=(14, 6))

    # Plot 1: Training & Validation Loss
    plt.subplot(1, 2, 1)
    plt.plot(epoch_range, train_loss, 'bo-', label='Training Loss')
    plt.plot(epoch_range, val_loss, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot 2: Validation Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epoch_range, val_acc, 'go-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    save_path = os.path.join(checkpoint_dir, 'training_history.png')
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"Using device: {device}")

    # datasets
    transform = default_transform
    if not os.path.exists(args.data_dir):
        alt = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'psg'))
        if os.path.exists(alt):
            args.data_dir = alt
        else:
            raise FileNotFoundError(f"data_dir '{args.data_dir}' not found and fallback '{alt}' also missing")

    train_ds = EEGDataset(data_dir=args.data_dir, split='train', fold=args.fold, window_sec=args.window_sec,
                          transform=transform, random_offset=True)
    val_ds = EEGDataset(data_dir=args.data_dir, split='val', fold=args.fold, window_sec=args.window_sec,
                        transform=transform, random_offset=False)
    
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("Training or validation dataset is empty. Check data directory and labels.csv")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)
    
    print(f"Training windows: {len(train_ds)}, Validation windows: {len(val_ds)}")

    if args.num_classes is None:
        if hasattr(train_ds, 'label_map') and isinstance(train_ds.label_map, dict):
            inferred = len(train_ds.label_map)
            print(f"Inferring num_classes={inferred} from dataset label_map")
            args.num_classes = inferred
        else:
            print("Warning: Could not infer num_classes, defaulting to 2.")
            args.num_classes = 2

    # model
    print(f"Loading model: resnet18_1d (num_classes={args.num_classes})")
    model = resnet18_1d(in_channels=len(train_ds.TARGET_CHANNELS), num_classes=args.num_classes)
    model.to(device)

    # Ambil nilai channels dan samples dari dataset
    n_channels = len(train_ds.TARGET_CHANNELS)
    n_samples = train_ds.sample_len 
    input_size = (args.batch_size, n_channels, n_samples)

    print("\n--- Model Architecture ---")
    summary(model, input_size=input_size)
    print("----------------------------\n")

    criterion = nn.CrossEntropyLoss()
    
    print(f"Using Adam optimizer with lr={args.lr} and weight_decay=1e-4")
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    best_val_loss = float('inf')

    sample_labels = [l for _, l in train_ds.data if l is not None]
    if len(sample_labels) > 0:
        max_label = max(sample_labels)
        if max_label >= args.num_classes:
            raise ValueError(f"Found label value {max_label} >= num_classes ({args.num_classes}).\n"
                             f"Check datareader.py logic and labels.csv mapping.")
    else:
        print("Warning: No labels found in training data for sanity check.")

    history_train_loss = []
    history_val_loss = []
    history_val_acc = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        total = 0

        for signals, labels in train_loader:
            signals = signals.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * signals.size(0)
            total += signals.size(0)

        train_loss = running_loss / max(total, 1)

        # validation
        model.eval()
        val_loss = 0.0
        val_total = 0
        correct = 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals = signals.to(device)
                labels = labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * signals.size(0)
                val_total += signals.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()

        val_loss = val_loss / max(val_total, 1)
        val_acc = correct / max(val_total, 1)

        print(f"Epoch {epoch}/{args.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        history_train_loss.append(train_loss)
        history_val_loss.append(val_loss)
        history_val_acc.append(val_acc)

        # save checkpoint
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        ckpt_path = os.path.join(args.checkpoint_dir, f"model_epoch{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, ckpt_path)

        if val_loss < best_val_loss:
            print(f"  Validation loss decreased ({best_val_loss:.4f} --> {val_loss:.4f}). Saving best model...")
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)

    print("Training finished. Generating plots...")
    plot_history(args.checkpoint_dir, args.epochs, history_train_loss, history_val_loss, history_val_acc)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='psg', help='Directory with EDF files')
    parser.add_argument('--epochs', type=int, default=10, help="Jumlah epochs (dinaikkan ke 15)")
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--num-workers', type=int, default=0, help='DataLoader num_workers (0 on Windows recommended)')
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--window-sec', type=int, default=5)
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number of output classes; if omitted the script will infer from labels.csv')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--no-cuda', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train(args)