import os
import torch
import numpy as np
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
from thop import profile, clever_format
from sklearn.metrics import confusion_matrix

# Mengimpor modul kustom - Pastikan datareader dan losses sudah lu update ke versi regresi!
from datareader import EEGDataset, collate_fn
from models import MambaDrowsinessDetector
from losses import compute_regression_metrics, get_classification_stats # Fungsi baru kita
from config import Config

def evaluate_model_complexity(model, device, input_shape=(1, 7, 15360)):
    print("\n" + "="*60)
    print("📊 EVALUASI KOMPLEKSITAS KOMPUTASI")
    print("="*60)
    
    dummy_input = torch.randn(input_shape).to(device)
    
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    gflops = macs / 1e9
    params_million = params / 1e6
    macs_str, params_str = clever_format([macs, params], "%.3f")
    
    print(f"\n📈 Hasil Evaluasi:")
    print(f"   ├─ Jumlah Parameter: {params_str} ({params_million:.2f}M)")
    print(f"   ├─ MACs: {macs_str}")
    print(f"   └─ GFLOPs: {gflops:.3f}")
    
    print(f"\n📝 Interpretasi:")
    print(f"   • Parameter count mengindikasikan ukuran model (memory footprint)")
    print(f"   • GFLOPs mengindikasikan kompleksitas komputasi (inference speed)")
    
    print(f"\n🔍 Perbandingan Kontekstual:")
    if params_million < 10.0:
        print(f"   ✅ Model ringan (1-10M parameters)")
    elif params_million < 50.0:
        print(f"   ⚠️  Model medium (10-50M parameters)")
    else:
        print(f"   ❌ Model berat (>50M parameters)")
    
    if gflops < 1.0:
        print(f"   ✅ Kompleksitas rendah (<1 GFLOPs)")
    else:
        print(f"   ⚠️  Kompleksitas tinggi")
    
    print("="*60 + "\n")
    return gflops, params_million, macs_str, params_str

def plot_confusion_matrix(y_true, y_pred, epoch, fold, phase='val', save_dir='confusion_matrices'):
    if torch.is_tensor(y_true): y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred): y_pred = y_pred.cpu().numpy()
    
    # Karena biner (threshold 5.5), kita pake label Alert vs Drowsy
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    class_names = ['Alert', 'Drowsy']
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, square=True, ax=ax)
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    ax.set_title(f'Confusion Matrix (Biner) - Fold {fold} - Epoch {epoch} ({phase.upper()})', fontsize=14, fontweight='bold', pad=20)
    
    accuracy = np.trace(cm) / np.sum(cm)
    fig.text(0.5, 0.02, f'Overall Accuracy (Semu): {accuracy:.2%} | Total Samples: {np.sum(cm)}', ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'cm_fold{fold}_epoch{epoch}_{phase}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if Config.USE_WANDB:
        wandb.log({f'confusion_matrix/{phase}_fold{fold}': wandb.Image(fig), 'epoch': epoch})
    plt.close(fig)
    return cm

def print_classification_report(cm):
    # Disederhanakan untuk 2 kelas (Biner)
    class_names = ['Alert', 'Drowsy']
    print("\n" + "="*70)
    print("📋 CLASSIFICATION REPORT (BINARY FROM REGRESSION)")
    print("="*70)
    print(f"{'Class':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12}")
    print("-"*70)
    
    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        print(f"{name:<20} {precision:>12.4f} {recall:>12.4f} {f1:>12.4f}")
    
    accuracy = np.trace(cm) / np.sum(cm)
    print("-"*70)
    print(f"{'Overall Accuracy':<20} {accuracy:>12.4f}")
    print("="*70 + "\n")

def train(fold=0):
    device = torch.device("cuda" if Config.USE_CUDA and torch.cuda.is_available() else "cpu")
    current_fold = fold
    
    print("\n" + "="*70)
    print(f"🚀 TRAINING REGRESI MAE FOLD {current_fold + 1}/{Config.N_SPLITS}")
    print("="*70)
    
    if Config.USE_WANDB:
        clean_config = {k: v for k, v in Config.to_dict().items() if not callable(v)}
        wandb.init(project=Config.WANDB_PROJECT, name=f"Reg_Fold_{current_fold}", config=clean_config, reinit=True)

    # --- 3. Persiapan Dataset & Dataloader ---
    train_dataset = EEGDataset(data_dir=Config.DATA_DIR, csv_path='label/labels.csv', fold=fold, split='train', n_splits=Config.N_SPLITS, use_augmentation=Config.USE_AUGMENTATION)
    val_dataset = EEGDataset(data_dir=Config.DATA_DIR, csv_path='label/labels.csv', fold=fold, split='val', n_splits=Config.N_SPLITS, use_augmentation=False)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=Config.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=Config.NUM_WORKERS, pin_memory=True)

    # --- 5. Inisialisasi Model ---
    model = MambaDrowsinessDetector(in_channels=Config.IN_CHANNELS, num_classes=1).to(device) # Regresi: 1 output neuron

    input_shape = (1, Config.IN_CHANNELS, Config.WINDOW_SEC * Config.SAMPLE_RATE)
    evaluate_model_complexity(model, device, input_shape)

    # --- 6. Optimizer & Loss ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = torch.nn.L1Loss() # MAE Loss untuk regresi
    
    scheduler = None
    if Config.USE_SCHEDULER:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=Config.SCHEDULER_FACTOR, patience=Config.SCHEDULER_PATIENCE, verbose=True)

    # --- 7. Loop Pelatihan ---
    best_val_mae = float('inf') # Nyari error terkecil
    patience_counter = 0
    
    for epoch in range(Config.EPOCHS):
        # FASE TRAINING
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
        for signals, labels in pbar:
            signals, labels = signals.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            total_train_loss += loss.item()
            pbar.set_postfix({"MAE": f"{loss.item():.4f}"})

        # FASE VALIDASI
        model.eval()
        val_preds_list, val_targets_list = [], []
        total_val_loss = 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                outputs = model(signals)
                total_val_loss += criterion(outputs, labels).item()
                val_preds_list.append(outputs.cpu()); val_targets_list.append(labels.cpu())

        # ========================================
        # PERHITUNGAN METRIK
        # ========================================
        val_preds = torch.cat(val_preds_list)
        val_targets = torch.cat(val_targets_list)
        
        # Metrik Regresi Skala 1-9
        val_mae, val_rmse = compute_regression_metrics(val_preds, val_targets)
        # Akurasi Semu (Thresholding 5.5)
        val_acc, preds_bin, targets_bin = get_classification_stats(val_preds, val_targets, threshold=5.5)
        
        avg_val_loss = total_val_loss / len(val_loader)
        if scheduler: scheduler.step(avg_val_loss)

        if Config.USE_WANDB:
            # FILTERING LEBIH KETAT: 
            # Cuma ambil variabel yang isinya angka, string, atau list (JSON friendly)
            raw_config = Config.to_dict()
            clean_config = {}
            for k, v in raw_config.items():
                # Cek apakah value-nya bukan fungsi, bukan classmethod, dan bisa di-serialize
                if not callable(v) and not isinstance(v, (classmethod, staticmethod)):
                    clean_config[k] = v

            wandb.init(
                project=Config.WANDB_PROJECT, 
                name=f"Reg_Fold_{current_fold}", 
                config=clean_config, 
                reinit=True
            )

        print(f"\n📊 Epoch {epoch+1} Summary: MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | Acc Semu: {val_acc:.4f}")

        # CONFUSION MATRIX
        if (epoch + 1) % 5 == 0 or epoch == Config.EPOCHS - 1:
            cm = plot_confusion_matrix(targets_bin, preds_bin, epoch+1, current_fold)
            print_classification_report(cm)

        # SIMPAN MODEL TERBAIK & EARLY STOPPING
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            checkpoint = {
                'epoch': epoch + 1, 'fold': current_fold, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'val_mae': val_mae, 'config': Config.to_dict()
            }
            torch.save(checkpoint, f"best_mamba_fold{current_fold}.pt")
            print(f"✅ Model terbaik disimpan (MAE: {val_mae:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}"); break

    # EVALUASI FINAL DENGAN BEST MODEL
    print("\n" + "="*60)
    print("🏁 PELATIHAN SELESAI - EVALUASI BEST MODEL")
    print("="*60)
    checkpoint = torch.load(f"best_mamba_fold{current_fold}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    final_preds, final_targets = [], []
    with torch.no_grad():
        for signals, labels in val_loader:
            outputs = model(signals.to(device))
            final_preds.append(outputs.cpu()); final_targets.append(labels.cpu())
    
    f_preds = torch.cat(final_preds); f_targets = torch.cat(final_targets)
    f_mae, _ = compute_regression_metrics(f_preds, f_targets)
    f_acc, f_p_bin, f_t_bin = get_classification_stats(f_preds, f_targets)
    
    print(f"✅ Final Best MAE: {f_mae:.4f}")
    print(f"✅ Final Accuracy Semu: {f_acc:.4f}")
    final_cm = plot_confusion_matrix(f_t_bin, f_p_bin, 'FINAL', current_fold, phase='val_best_model')
    print_classification_report(final_cm)

    if Config.USE_WANDB: wandb.finish()
    return best_val_mae

if __name__ == "__main__":
    train(fold=0)