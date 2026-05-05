import os
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from datareader import EEGDataset, collate_fn
from models import MambaDrowsinessDetector
from losses import MAELoss, WeightedCrossEntropyLoss, compute_inverse_weight, compute_regression_metrics, get_classification_stats
from config import Config
from utils import (
    evaluate_model_complexity, plot_confusion_matrix,
    print_classification_report, save_checkpoint, compute_epoch_metrics,
    get_class_names
)

def _make_loader(fold, split, augment):
    ds = EEGDataset(
        data_dir=Config.DATA_DIR,
        csv_path='label/labels.csv',
        fold=fold, split=split,
        n_splits=Config.N_SPLITS,
        window_sec=Config.WINDOW_SEC,
        stride_sec=Config.STRIDE_SEC if split == 'train' else Config.WINDOW_SEC,
        use_augmentation=augment
    )
    return DataLoader(
        ds, batch_size=Config.BATCH_SIZE,
        shuffle=(split == 'train'),
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=(split == 'train')
    ), ds


def _run_epoch(model, loader, criterion, optimizer, device, is_train):
    model.train() if is_train else model.eval()
    total_loss, preds_list, targets_list = 0.0, [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    tag = "Train" if is_train else "Val"
    with ctx:
        for signals, labels in tqdm(loader, desc=tag, leave=False):
            signals, labels = signals.to(device), labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                optimizer.step()

            total_loss += loss.item()
            if Config.is_classification():
                preds_list.append(torch.argmax(outputs, dim=-1).detach().cpu())
            else:
                preds_list.append(outputs.detach().cpu())
            targets_list.append(labels.detach().cpu())

    preds = torch.cat(preds_list)
    targets = torch.cat(targets_list)
    avg_loss = total_loss / len(loader)
    acc, f1, mae, rmse, preds_b, targets_b = compute_epoch_metrics(preds, targets)
    return avg_loss, acc, f1, mae, rmse, preds_b, targets_b


def _build_scheduler(optimizer):
    if not Config.USE_SCHEDULER:
        return None, None
    from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR
    plateau = ReduceLROnPlateau(optimizer, mode='min',
                                factor=Config.SCHEDULER_FACTOR,
                                patience=Config.SCHEDULER_PATIENCE, verbose=True)
    use_warmup = getattr(Config, 'USE_WARMUP', False)
    warmup_epochs = getattr(Config, 'WARMUP_EPOCHS', 0)
    warmup = LinearLR(optimizer, start_factor=getattr(Config, 'WARMUP_START_FACTOR', 0.1),
                      end_factor=1.0, total_iters=warmup_epochs) if use_warmup and warmup_epochs > 0 else None
    return warmup, plateau


def _maybe_save_cm(epoch, fold, tr_b, tr_t, val_b, val_t, save_dir, interval=5):
    if (epoch + 1) % interval != 0 and epoch != Config.EPOCHS - 1:
        return
    class_names = get_class_names()
    labels = list(range(len(class_names)))
    plot_confusion_matrix(tr_t, tr_b, epoch+1, fold, 'train', save_dir, labels, class_names)
    val_cm = plot_confusion_matrix(val_t, val_b, epoch+1, fold, 'val', save_dir, labels, class_names)
    print_classification_report(val_cm, class_names)


def _final_eval(model, val_loader, device, fold, save_dir):
    print("Generating final confusion matrix with best model...")
    ckpt = torch.load(f"best_mamba_fold{fold}.pt")
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    preds_list, targets_list = [], []
    with torch.no_grad():
        for signals, labels in val_loader:
            preds_list.append(model(signals.to(device)).detach().cpu())
            targets_list.append(labels.cpu())

    preds = torch.cat(preds_list)
    targets = torch.cat(targets_list)
    acc, f1, mae, rmse, preds_b, targets_b = compute_epoch_metrics(preds, targets)

    class_names = get_class_names()
    labels = list(range(len(class_names)))
    final_cm = plot_confusion_matrix(targets_b, preds_b, 'final', fold,
                                     'val_best_model', save_dir, labels, class_names)
    print_classification_report(final_cm, class_names)

    if Config.is_classification():
        print(f"Final Best Model — Acc: {acc:.4f} | F1: {f1:.4f}")
    else:
        print(f"Final Best Model — MAE: {mae:.4f} | RMSE: {rmse:.4f} | Acc(thr): {acc:.4f}")


def train(fold=0):
    device = torch.device("cuda" if Config.USE_CUDA and torch.cuda.is_available() else "cpu")
    cm_save_dir = 'confusion_matrices'
    print(f"\n{'='*60}\n🚀 TRAINING FOLD {fold+1}/{Config.N_SPLITS} | device={device}\n{'='*60}")

    # WandB
    if Config.USE_WANDB:
        raw = Config.to_dict()
        clean = {k: v for k, v in raw.items()
                 if not k.startswith('__') and not isinstance(v, (classmethod, staticmethod)) and not callable(v)}
        wandb.init(project=Config.WANDB_PROJECT, name=f"fold_{fold}_{Config.TASK_TYPE}",
                   config=clean, reinit=True)

    # Data
    train_loader, train_ds = _make_loader(fold, 'train', Config.USE_AUGMENTATION)
    val_loader, _          = _make_loader(fold, 'val',   False)

    # Model
    model = MambaDrowsinessDetector(
        in_channels=Config.IN_CHANNELS, num_classes=Config.get_output_dim(),
        d_model=Config.MAMBA_D_MODEL, n_layers=Config.MAMBA_N_LAYERS,
        d_state=Config.MAMBA_D_STATE, d_conv=Config.MAMBA_D_CONV, expand=Config.MAMBA_EXPAND
    ).to(device)

    input_shape = (1, Config.IN_CHANNELS, Config.WINDOW_SEC * Config.SAMPLE_RATE)
    gflops, params_m, macs_str, params_str = evaluate_model_complexity(model, device, input_shape)

    if Config.USE_WANDB:
        wandb.config.update({"model_gflops": gflops, "model_params_million": params_m,
                             "model_params_str": params_str, "model_macs_str": macs_str})

    # Optimizer, Loss, Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE,
                                  weight_decay=Config.WEIGHT_DECAY)
    if Config.is_classification():
        weights = compute_inverse_weight([s['class_idx'] for s in train_ds.samples],
                                         num_classes=Config.NUM_CLASSES)
        criterion = WeightedCrossEntropyLoss(weight=weights.to(device))
    else:
        criterion = MAELoss()

    warmup_sched, plateau_sched = _build_scheduler(optimizer)
    warmup_epochs = getattr(Config, 'WARMUP_EPOCHS', 0)

    # Best-metric tracking
    best = {'acc': 0.0, 'f1': 0.0} if Config.is_classification() else {'mae': float('inf'), 'rmse': float('inf')}
    patience_counter = 0

    # ── Training Loop ───────────────────────────────────────────────────────
    for epoch in range(Config.EPOCHS):
        tr_loss, tr_acc, tr_f1, tr_mae, tr_rmse, tr_pb, tr_tb = _run_epoch(
            model, train_loader, criterion, optimizer, device, is_train=True)
        val_loss, val_acc, val_f1, val_mae, val_rmse, val_pb, val_tb = _run_epoch(
            model, val_loader, criterion, optimizer, device, is_train=False)

        # Scheduler step
        if plateau_sched:
            if warmup_sched and epoch < warmup_epochs:
                warmup_sched.step()
            plateau_sched.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # Log
        if Config.USE_WANDB:
            wandb.log({"epoch": epoch+1, "lr": current_lr,
                       "train/loss": tr_loss, "train/acc": tr_acc, "train/f1": tr_f1,
                       "train/mae": tr_mae, "train/rmse": tr_rmse,
                       "val/loss": val_loss, "val/acc": val_acc, "val/f1": val_f1,
                       "val/mae": val_mae, "val/rmse": val_rmse})

        print(f"Epoch {epoch+1:>3}/{Config.EPOCHS} | "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} f1={tr_f1:.4f} | "
              f"val loss={val_loss:.4f} acc={val_acc:.4f} f1={val_f1:.4f} | lr={current_lr:.2e}")

        # Confusion matrix (periodic)
        _maybe_save_cm(epoch, fold, tr_pb, tr_tb, val_pb, val_tb, cm_save_dir)

        # Best model + early stopping
        improved = (val_acc > best['acc']) if Config.is_classification() else (val_mae < best['mae'])
        if improved:
            patience_counter = 0
            if Config.is_classification():
                best['acc'], best['f1'] = val_acc, val_f1
                metrics = {'val_acc': val_acc, 'val_f1': val_f1}
            else:
                best['mae'], best['rmse'] = val_mae, val_rmse
                metrics = {'val_mae': val_mae, 'val_rmse': val_rmse}
            if Config.SAVE_BEST_ONLY:
                save_checkpoint(model, optimizer, epoch+1, fold, metrics)
        else:
            patience_counter += 1

        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"⚠️  Early stopping at epoch {epoch+1}")
            break

    # Summary
    print(f"\n🏁 Training done | Best: {best}")
    if Config.USE_WANDB:
        wandb.run.summary.update({f"best_{k}": v for k, v in best.items()})

    # Final eval with best checkpoint
    _final_eval(model, val_loader, device, fold, cm_save_dir)

    if Config.USE_WANDB:
        wandb.finish()

    return best


if __name__ == "__main__":
    results = []
    for fold in range(Config.N_SPLITS):
        print(f"\n{'='*60}\n📂 FOLD {fold+1}/{Config.N_SPLITS}\n{'='*60}")
        result = train(fold=fold)
        result['fold'] = fold
        results.append(result)

    print(f"\n{'='*60}\n📊 5-FOLD CV RESULTS\n{'='*60}")
    keys = [k for k in results[0] if k != 'fold']
    for r in results:
        print("Fold", r['fold']+1, "|", " | ".join(f"{k}={r[k]:.4f}" for k in keys))
    for k in keys:
        vals = [r[k] for r in results]
        print(f"Mean {k}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    print("=" * 60)