import os
import argparse
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datareader import EEGDataset
from models import resnet34_1d


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


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')

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

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)

    # infer num_classes from dataset if user didn't pass one
    if args.num_classes is None:
        if hasattr(train_ds, 'label_map') and isinstance(train_ds.label_map, dict):
            inferred = len(train_ds.label_map)
            print(f"Inferring num_classes={inferred} from dataset label_map")
            args.num_classes = inferred
        else:
            args.num_classes = 2

    # model
    model = resnet34_1d(in_channels=len(train_ds.TARGET_CHANNELS), num_classes=args.num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float('inf')
    sample_labels = [l for l in train_ds.labels if l is not None]
    if len(sample_labels) > 0:
        max_label = max(sample_labels)
        if max_label >= args.num_classes:
            raise ValueError(f"Found label value {max_label} >= num_classes ({args.num_classes}).\n"
                             f"Check labels.csv mapping or pass --num-classes accordingly.")

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
            best_val_loss = val_loss
            best_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='psg', help='Directory with EDF files')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
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
