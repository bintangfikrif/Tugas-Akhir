import os
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from thop import profile
from sklearn.metrics import confusion_matrix, f1_score
from losses import compute_regression_metrics, get_classification_stats
from config import Config

CLASS_NAMES = {
    2: ['Alert', 'Drowsy'],
    3: ['Alert', 'Low Vigilance', 'Drowsy']
}

def get_class_names():
    return CLASS_NAMES.get(Config.NUM_CLASSES, ['Alert', 'Drowsy'])

# ── Model Complexity ──────────────────────────────────────────────────────────

def evaluate_model_complexity(model, device, input_shape=(1, 7, 15360)):
    dummy_input = torch.randn(input_shape).to(device)
    model.eval()
    with torch.no_grad():
        macs_thop, _ = profile(model, inputs=(dummy_input,), verbose=False)
        total_params = model.get_num_params()
        L, D, N = 480, Config.MAMBA_D_MODEL, Config.MAMBA_D_STATE
        macs_mamba = (9 * 1 * L * D * N * Config.MAMBA_N_LAYERS) / 2
        total_macs = macs_thop + macs_mamba

    gflops = total_macs / 1e9
    params_million = total_params / 1e6
    macs_str = f"{total_macs / 1e6:.3f}M"
    params_str = f"{total_params / 1e3:.3f}K"
    print(f"[Complexity] Params: {params_str} | MACs: {macs_str} | GFLOPs: {gflops:.6f}")
    return gflops, params_million, macs_str, params_str

# ── Confusion Matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, epoch, fold, phase='val',
                          save_dir='confusion_matrices', labels=None, class_names=None):
    if torch.is_tensor(y_true): y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred): y_pred = y_pred.cpu().numpy()

    class_names = class_names or get_class_names()
    labels = labels or list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                square=True, linewidths=0.5, linecolor='gray', ax=ax)
    ax.set_xlabel('Predicted', fontweight='bold')
    ax.set_ylabel('Actual', fontweight='bold')
    ax.set_title(f'CM - Fold {fold} - Epoch {epoch} ({phase.upper()})', fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'cm_fold{fold}_epoch{epoch}_{phase}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    wandb.log({f'confusion_matrix/{phase}_fold{fold}': wandb.Image(fig), 'epoch': epoch})
    plt.close(fig)
    return cm

# ── Classification Report ─────────────────────────────────────────────────────

def print_classification_report(cm, class_names=None):
    class_names = class_names or get_class_names()
    print(f"\n{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 60)
    total, wp, wr, wf = 0, 0, 0, 0
    for i, name in enumerate(class_names):
        tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i, :].sum() - tp
        support = tp + fn
        p = tp / (tp + fp + 1e-6); r = tp / (tp + fn + 1e-6)
        f = 2 * p * r / (p + r + 1e-6)
        total += support; wp += p * support; wr += r * support; wf += f * support
        print(f"{name:<20} {p:>10.4f} {r:>10.4f} {f:>10.4f} {int(support):>10}")
    print("-" * 60)
    print(f"{'Weighted Avg':<20} {wp/total:>10.4f} {wr/total:>10.4f} {wf/total:>10.4f} {int(total):>10}")
    print(f"{'Accuracy':<20} {np.trace(cm)/np.sum(cm):>10.4f}\n")

# ── Checkpoint ───────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, fold, metrics: dict):
    raw = Config.to_dict()
    clean_config = {k: v for k, v in raw.items()
                    if not isinstance(v, (classmethod, staticmethod)) and not callable(v)}
    path = f"best_mamba_fold{fold}.pt"
    torch.save({
        'epoch': epoch, 'fold': fold,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': clean_config, **metrics
    }, path)
    if Config.USE_WANDB:
        wandb.save(path)
    print(f"✅ Checkpoint saved: {path} | " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

# ── Metric Computation ────────────────────────────────────────────────────────

def compute_epoch_metrics(preds, targets):
    """Returns (acc, f1, mae, rmse, preds_binary, targets_binary)."""
    is_cls = Config.is_classification()
    if is_cls:
        preds_b, targets_b = preds, targets
        acc = (preds == targets).float().mean().item()
        f1 = f1_score(targets, preds, average='macro', zero_division=0)
        mae, rmse = 0.0, 0.0
    else:
        preds, targets = preds.squeeze(-1), targets.squeeze(-1)
        mae, rmse = compute_regression_metrics(preds, targets)
        acc, preds_b, targets_b = get_classification_stats(preds, targets)
        f1 = f1_score(targets_b, preds_b, average='binary', zero_division=0)
    return acc, f1, mae, rmse, preds_b, targets_b