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

# Mengimpor modul kustom yang telah disesuaikan dengan proposal
from datareader import EEGDataset, collate_fn, OneSamplePerRecordingSampler
from models import MambaDrowsinessDetector
from losses import WeightedCrossEntropyLoss, compute_inverse_weight, get_evaluation_metrics
from config import Config

def evaluate_model_complexity(model, device, input_shape=(1, 7, 15360)):
    print("\n" + "="*60)
    print("📊 EVALUASI KOMPLEKSITAS KOMPUTASI")
    print("="*60)
    
    # Buat dummy input sesuai format data EEG
    dummy_input = torch.randn(input_shape).to(device)
    
    # Hitung FLOPs dan Parameters menggunakan thop
    model.eval()
    with torch.no_grad():
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
    
    # Convert ke format human-readable
    gflops = macs / 1e9  # Convert ke Giga FLOPs
    params_million = params / 1e6  # Convert ke Millions
    
    # Format dengan clever_format untuk output yang rapi
    macs_str, params_str = clever_format([macs, params], "%.3f")
    
    print(f"\n📈 Hasil Evaluasi:")
    print(f"   ├─ Jumlah Parameter: {params_str} ({params_million:.2f}M)")
    print(f"   ├─ MACs: {macs_str}")
    print(f"   └─ GFLOPs: {gflops:.3f}")
    
    print(f"\n📝 Interpretasi:")
    print(f"   • Parameter count mengindikasikan ukuran model (memory footprint)")
    print(f"   • GFLOPs mengindikasikan kompleksitas komputasi (inference speed)")
    
    print(f"\n🔍 Perbandingan Kontekstual:")
    if params_million < 1.0:
        print(f"   ✅ Model sangat ringan (<1M parameters)")
    elif params_million < 10.0:
        print(f"   ✅ Model ringan (1-10M parameters)")
    elif params_million < 50.0:
        print(f"   ⚠️  Model medium (10-50M parameters)")
    else:
        print(f"   ❌ Model berat (>50M parameters)")
    
    if gflops < 1.0:
        print(f"   ✅ Kompleksitas rendah (<1 GFLOPs)")
    elif gflops < 5.0:
        print(f"   ✅ Kompleksitas medium (1-5 GFLOPs)")
    else:
        print(f"   ⚠️  Kompleksitas tinggi (>5 GFLOPs)")
    
    print("="*60 + "\n")
    
    return gflops, params_million, macs_str, params_str

def plot_confusion_matrix(y_true, y_pred, epoch, fold, phase='val', save_dir='confusion_matrices'):
    # Convert tensor to numpy jika perlu
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()
    
    # Hitung confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Class names sesuai proposal (hal. 32)
    class_names = ['Alert', 'Low Vigilance', 'Drowsy']
    
    # Buat figure dengan ukuran yang sesuai
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot heatmap menggunakan seaborn untuk visualisasi yang lebih baik
    sns.heatmap(
        cm, 
        annot=True,           # Tampilkan angka di setiap sel
        fmt='d',              # Format integer
        cmap='Blues',         # Color map biru sesuai standar akademik
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Number of Samples'},
        square=True,          # Buat sel berbentuk persegi
        linewidths=0.5,       # Border antar sel
        linecolor='gray',
        ax=ax
    )
    
    # Konfigurasi label dan title
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual Label', fontsize=12, fontweight='bold')
    ax.set_title(
        f'Confusion Matrix - Fold {fold} - Epoch {epoch} ({phase.upper()})',
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    
    # Rotasi label untuk keterbacaan lebih baik
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    plt.setp(ax.get_yticklabels(), rotation=0)
    
    # Tambahkan informasi metrik di bawah plot
    accuracy = np.trace(cm) / np.sum(cm)
    fig.text(
        0.5, 0.02,
        f'Overall Accuracy: {accuracy:.2%} | Total Samples: {np.sum(cm)}',
        ha='center',
        fontsize=10,
        style='italic'
    )
    
    plt.tight_layout()
    
    # Simpan ke file lokal
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'cm_fold{fold}_epoch{epoch}_{phase}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Log ke WandB
    wandb.log({
        f'confusion_matrix/{phase}_fold{fold}': wandb.Image(fig),
        'epoch': epoch
    })
    
    plt.close(fig)
    
    print(f"📊 Confusion Matrix disimpan: {save_path}")
    
    return cm

def compute_per_class_metrics(cm, class_names=['Alert', 'Low Vigilance', 'Drowsy']):
    metrics = {}
    
    for i, class_name in enumerate(class_names):
        # True Positive: diagonal element
        tp = cm[i, i]
        
        # False Positive: sum of column excluding diagonal
        fp = cm[:, i].sum() - tp
        
        # False Negative: sum of row excluding diagonal
        fn = cm[i, :].sum() - tp
        
        # True Negative: sum of all other elements
        tn = cm.sum() - tp - fp - fn
        
        # Hitung metrik sesuai rumus 2.4, 2.5, 2.6, 2.7 (hal. 25-26)
        precision = tp / (tp + fp + 1e-6)
        recall = tp / (tp + fn + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn)
        }
    
    return metrics

def print_classification_report(cm, class_names=['Alert', 'Low Vigilance', 'Drowsy']):
    """
    Print detailed classification report dari confusion matrix.
    """
    metrics = compute_per_class_metrics(cm, class_names)
    
    print("\n" + "="*70)
    print("📋 CLASSIFICATION REPORT")
    print("="*70)
    print(f"{'Class':<20} {'Precision':>12} {'Recall':>12} {'F1-Score':>12} {'Support':>10}")
    print("-"*70)
    
    total_support = 0
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0
    
    for class_name in class_names:
        m = metrics[class_name]
        support = m['tp'] + m['fn']
        total_support += support
        
        weighted_precision += m['precision'] * support
        weighted_recall += m['recall'] * support
        weighted_f1 += m['f1_score'] * support
        
        print(f"{class_name:<20} {m['precision']:>12.4f} {m['recall']:>12.4f} {m['f1_score']:>12.4f} {support:>10d}")
    
    print("-"*70)
    
    # Macro average (simple average)
    macro_precision = np.mean([m['precision'] for m in metrics.values()])
    macro_recall = np.mean([m['recall'] for m in metrics.values()])
    macro_f1 = np.mean([m['f1_score'] for m in metrics.values()])
    
    print(f"{'Macro Avg':<20} {macro_precision:>12.4f} {macro_recall:>12.4f} {macro_f1:>12.4f} {total_support:>10d}")
    
    # Weighted average
    weighted_precision /= total_support
    weighted_recall /= total_support
    weighted_f1 /= total_support
    
    print(f"{'Weighted Avg':<20} {weighted_precision:>12.4f} {weighted_recall:>12.4f} {weighted_f1:>12.4f} {total_support:>10d}")
    
    # Overall accuracy
    accuracy = np.trace(cm) / np.sum(cm)
    print("-"*70)
    print(f"{'Overall Accuracy':<20} {accuracy:>12.4f}")
    print("="*70 + "\n")
    
    return metrics

def train(fold=0):  # ✅ TAMBAHKAN parameter fold
    # --- 1. Konfigurasi Eksperimen ---
    device = torch.device("cuda" if Config.USE_CUDA and torch.cuda.is_available() else "cpu")
    
    # ✅ GUNAKAN Config untuk semua parameter
    window_sec = Config.WINDOW_SEC
    n_splits = Config.N_SPLITS
    batch_size = Config.BATCH_SIZE
    epochs = Config.EPOCHS
    lr = Config.LEARNING_RATE
    current_fold = fold  # 
    
    # Konfigurasi Confusion Matrix
    cm_save_interval = 5
    cm_save_dir = 'confusion_matrices'
    
    print("\n" + "="*70)
    print(f"🚀 TRAINING FOLD {current_fold + 1}/{n_splits}")
    print("="*70)
    print(f"📊 Konfigurasi:")
    print(f"   ├─ Device: {device}")
    print(f"   ├─ Window: {window_sec}s")
    print(f"   ├─ Batch Size: {batch_size}")
    print(f"   ├─ Learning Rate: {lr}")
    print(f"   ├─ Epochs: {epochs}")
    print(f"   ├─ Data Augmentation: {Config.USE_AUGMENTATION}")  # ✅ Log augmentation status
    print("="*70)
    
# --- 2. Inisialisasi WandB ---
    if Config.USE_WANDB:
        # Ambil config mentah
        raw_config = Config.to_dict()
        
        # FILTER: Buang item yang berupa function atau classmethod agar tidak error
        clean_config = {
            k: v for k, v in raw_config.items() 
            if not k.startswith('__') and not isinstance(v, (classmethod, staticmethod)) and not callable(v)
        }

        wandb.init(
            project=Config.WANDB_PROJECT,
            name=f"Mamba_Fold_{current_fold}_OneSamplePerRecording",
            config=clean_config,  
            reinit=True  
        )

    # --- 3. Persiapan Dataset & Dataloader ---
    train_dataset = EEGDataset(
        data_dir=Config.DATA_DIR,
        csv_path=os.path.join('label/labels.csv'),
        fold=fold,
        split='train',
        n_splits=Config.N_SPLITS,
        window_sec=Config.WINDOW_SEC,
        stride_sec=Config.WINDOW_SEC,   
        use_augmentation=True
    )           
    
    val_dataset = EEGDataset(
        data_dir=Config.DATA_DIR,
        csv_path=os.path.join('label/labels.csv'),
        fold=fold,
        split='val',
        n_splits=Config.N_SPLITS,
        window_sec=Config.WINDOW_SEC,
        stride_sec=Config.WINDOW_SEC,   
        use_augmentation=False          
    )

    custom_sampler = OneSamplePerRecordingSampler(train_dataset, Config.BATCH_SIZE)

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=custom_sampler,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )

    # --- 4. Penanganan Ketidakseimbangan Data ---
    train_labels = [item['label'] for item in train_dataset.samples]
    class_weights = compute_inverse_weight(train_labels, num_classes=Config.NUM_CLASSES).to(device)
    print(f"\nBobot Kelas (Fold {current_fold}): {class_weights}")

    # --- 5. Inisialisasi Model ---
    # ✅ GUNAKAN Config untuk parameter model
    model = MambaDrowsinessDetector(
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES,
        d_model=Config.MAMBA_D_MODEL,
        n_layers=Config.MAMBA_N_LAYERS,
        d_state=Config.MAMBA_D_STATE,
        d_conv=Config.MAMBA_D_CONV,
        expand=Config.MAMBA_EXPAND
    ).to(device)

    # ✅ Evaluasi Kompleksitas Komputasi
    input_shape = (1, Config.IN_CHANNELS, Config.WINDOW_SEC * Config.SAMPLE_RATE)
    gflops, params_million, macs_str, params_str = evaluate_model_complexity(
        model, device, input_shape
    )
    
    # Log ke WandB
    if Config.USE_WANDB:
        wandb.config.update({
            "model_gflops": gflops,
            "model_params_million": params_million,
            "model_params_str": params_str,
            "model_macs_str": macs_str
        })
        
        wandb.run.summary["total_params"] = params_str
        wandb.run.summary["gflops"] = f"{gflops:.3f}"

    # --- 6. Optimizer & Loss ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )
    
    criterion = WeightedCrossEntropyLoss(weight=class_weights)
    
    # TAMBAHKAN Learning Rate Scheduler 
    scheduler = None
    if Config.USE_SCHEDULER:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='max',  # Karena monitor accuracy
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            verbose=True
        )
        print(f"✅ Learning Rate Scheduler enabled (patience={Config.SCHEDULER_PATIENCE})")

    # --- 7. Loop Pelatihan ---
    best_val_acc = 0
    best_val_f1 = 0
    patience_counter = 0  # Untuk early stopping
    
    print("\n" + "="*60)
    print("MEMULAI PELATIHAN")
    print("="*60 + "\n")
    
    for epoch in range(Config.EPOCHS):
        # ========================================
        # FASE TRAINING
        # ========================================
        model.train()
        total_train_loss = 0
        train_preds_list = []
        train_targets_list = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Train]")
        for signals, labels in pbar:
            signals, labels = signals.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Simpan prediksi untuk confusion matrix
            preds = torch.argmax(logits, dim=1)
            train_preds_list.append(preds.cpu())
            train_targets_list.append(labels.cpu())
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # ========================================
        # FASE VALIDASI
        # ========================================
        model.eval()
        val_preds_list = []
        val_targets_list = []
        total_val_loss = 0
        
        with torch.no_grad():
            pbar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.EPOCHS} [Val]")
            for signals, labels in pbar_val:
                signals, labels = signals.to(device), labels.to(device)
                logits = model(signals)
                val_loss = criterion(logits, labels)
                total_val_loss += val_loss.item()
                
                preds = torch.argmax(logits, dim=1)
                val_preds_list.append(preds.cpu())
                val_targets_list.append(labels.cpu())
                
                pbar_val.set_postfix({"loss": f"{val_loss.item():.4f}"})

        # ========================================
        # PERHITUNGAN METRIK
        # ========================================
        train_preds = torch.cat(train_preds_list)
        train_targets = torch.cat(train_targets_list)
        val_preds = torch.cat(val_preds_list)
        val_targets = torch.cat(val_targets_list)
        
        train_acc, train_metrics = get_evaluation_metrics(train_preds, train_targets)
        val_acc, val_metrics = get_evaluation_metrics(val_preds, val_targets)
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        train_f1_macro = np.mean([train_metrics[f'class_{i}']['f1'].item() for i in range(Config.NUM_CLASSES)])
        val_f1_macro = np.mean([val_metrics[f'class_{i}']['f1'].item() for i in range(Config.NUM_CLASSES)])

        # Learning Rate Scheduler Step
        if scheduler is not None:
            scheduler.step(val_acc)
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = Config.LEARNING_RATE

        # ========================================
        # LOGGING KE WANDB
        # ========================================
        if Config.USE_WANDB:
            wandb.log({
                "epoch": epoch + 1,
                "learning_rate": current_lr,  # ✅ Log LR
                "train/loss": avg_train_loss,
                "train/accuracy": train_acc.item(),
                "train/f1_macro": train_f1_macro,
                "train/f1_alert": train_metrics['class_0']['f1'].item(),
                "train/f1_low_vigilance": train_metrics['class_1']['f1'].item(),
                "train/f1_drowsy": train_metrics['class_2']['f1'].item(),
                "val/loss": avg_val_loss,
                "val/accuracy": val_acc.item(),
                "val/f1_macro": val_f1_macro,
                "val/f1_alert": val_metrics['class_0']['f1'].item(),
                "val/f1_low_vigilance": val_metrics['class_1']['f1'].item(),
                "val/f1_drowsy": val_metrics['class_2']['f1'].item(),
            })

        # Print hasil epoch
        print(f"\n📊 Epoch {epoch+1}/{Config.EPOCHS} Summary:")
        print(f"   Train - Loss: {avg_train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1_macro:.4f}")
        print(f"   Val   - Loss: {avg_val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1_macro:.4f}")
        print(f"   LR: {current_lr:.2e}")

        # ========================================
        # CONFUSION MATRIX
        # ========================================
        if (epoch + 1) % cm_save_interval == 0 or epoch == Config.EPOCHS - 1:
            print(f"\n📈 Generating Confusion Matrix...")
            
            train_cm = plot_confusion_matrix(
                train_targets, train_preds,
                epoch=epoch+1, fold=current_fold,
                phase='train', save_dir=cm_save_dir
            )
            
            val_cm = plot_confusion_matrix(
                val_targets, val_preds,
                epoch=epoch+1, fold=current_fold,
                phase='val', save_dir=cm_save_dir
            )
            
            print(f"\n📋 Validation Set Classification Report:")
            val_class_metrics = print_classification_report(val_cm)

        # ========================================
        # SIMPAN MODEL TERBAIK & EARLY STOPPING
        # ========================================
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_f1 = val_f1_macro
            patience_counter = 0  # ✅ Reset counter
            
            if Config.SAVE_BEST_ONLY:
                model_name = f"best_mamba_fold{current_fold}.pt"
                raw_config = Config.to_dict()
                clean_config = {
                    k: v for k, v in raw_config.items() 
                    if not isinstance(v, (classmethod, staticmethod)) and not callable(v)
                }
                checkpoint = {
                    'epoch': epoch + 1,
                    'fold': current_fold,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc.item(),
                    'val_f1': val_f1_macro,
                    'class_weights': class_weights,
                    'config': clean_config
                }
                
                torch.save(checkpoint, model_name)
                if Config.USE_WANDB:
                    wandb.save(model_name)
                
                print(f"\n✅ Model terbaik disimpan: {model_name}")
                print(f"   └─ Val Acc: {val_acc:.4f} | Val F1: {val_f1_macro:.4f}\n")
        else:
            patience_counter += 1  # ✅ Increment counter
        
        # Early Stopping Check
        if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch+1}")
            print(f"   No improvement for {Config.EARLY_STOPPING_PATIENCE} epochs")
            break

    # ========================================
    # EVALUASI FINAL
    # ========================================
    print("\n" + "="*60)
    print("🏁 PELATIHAN SELESAI")
    print("="*60)
    print(f"✅ Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"✅ Best Validation F1-Score: {best_val_f1:.4f}")
    print("="*60 + "\n")
    
    if Config.USE_WANDB:
        wandb.run.summary["best_val_acc"] = best_val_acc.item()
        wandb.run.summary["best_val_f1"] = best_val_f1
    
    # Final confusion matrix dengan best model
    print("📊 Generating Final Confusion Matrix with Best Model...")
    checkpoint = torch.load(f"best_mamba_fold{current_fold}.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    final_val_preds = []
    final_val_targets = []
    
    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.to(device)
            logits = model(signals)
            preds = torch.argmax(logits, dim=1)
            final_val_preds.append(preds.cpu())
            final_val_targets.append(labels.cpu())
    
    final_val_preds = torch.cat(final_val_preds)
    final_val_targets = torch.cat(final_val_targets)
    
    final_cm = plot_confusion_matrix(
        final_val_targets, final_val_preds,
        epoch='final', fold=current_fold,
        phase='val_best_model', save_dir=cm_save_dir
    )
    
    print("\n📋 Final Model Classification Report:")
    print_classification_report(final_cm)

    if Config.USE_WANDB:
        wandb.finish()
    
    return best_val_acc.item(), best_val_f1

if __name__ == "__main__":
    # Train single fold
    # train(fold=0)
    
    # Fll 5-fold CV
    results = []
    for fold in range(Config.N_SPLITS):
        print(f"\n{'='*70}")
        print(f"📂 STARTING FOLD {fold + 1}/{Config.N_SPLITS}")
        print(f"{'='*70}\n")
        
        acc, f1 = train(fold=fold)
        results.append({'fold': fold, 'accuracy': acc, 'f1': f1})
        
        print(f"\n✅ Fold {fold + 1} completed: Acc={acc:.4f}, F1={f1:.4f}\n")
    
    # Ringkasan hasil
    print("\n" + "="*70)
    print("📊 5-FOLD CROSS VALIDATION RESULTS")
    print("="*70)
    for r in results:
        print(f"Fold {r['fold']+1}: Acc={r['accuracy']:.4f}, F1={r['f1']:.4f}")
    
    mean_acc = np.mean([r['accuracy'] for r in results])
    std_acc = np.std([r['accuracy'] for r in results])
    mean_f1 = np.mean([r['f1'] for r in results])
    std_f1 = np.std([r['f1'] for r in results])
    
    print("-"*70)
    print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Mean F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
    print("="*70)