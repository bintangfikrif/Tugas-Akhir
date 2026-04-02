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
from datareader import EEGDataset, collate_fn
from models import MambaDrowsinessDetector
from losses import MAELoss, WeightedCrossEntropyLoss, compute_inverse_weight, compute_regression_metrics, get_classification_stats
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

def plot_confusion_matrix(y_true, y_pred, epoch, fold, phase='val', save_dir='confusion_matrices', labels=None, class_names=None):
    # Convert tensor to numpy jika perlu
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.cpu().numpy()

    if labels is None:
        labels = [0, 1] if class_names is not None else list(range(Config.NUM_CLASSES))
    if class_names is None:
        if Config.NUM_CLASSES == 2:
            class_names = ['Alert', 'Drowsy']
        else:
            class_names = ['Alert', 'Low Vigilance', 'Drowsy']

    # Hitung confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    # Buat figure dengan ukuran yang sesuai
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot heatmap menggunakan seaborn untuk visualisasi yang lebih baik
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Number of Samples'},
        square=True,
        linewidths=0.5,
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

def compute_per_class_metrics(cm, class_names=None):
    if class_names is None:
        class_names = ['Alert', 'Drowsy'] if Config.NUM_CLASSES == 2 else ['Alert', 'Low Vigilance', 'Drowsy']
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

def print_classification_report(cm, class_names=None):
    """
    Print detailed classification report dari confusion matrix.
    """
    if class_names is None:
        class_names = ['Alert', 'Drowsy'] if Config.NUM_CLASSES == 2 else ['Alert', 'Low Vigilance', 'Drowsy']
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
            name=f"fold_{current_fold}_{Config.TASK_TYPE}",
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
        stride_sec=Config.STRIDE_SEC,   
        use_augmentation=Config.USE_AUGMENTATION
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

    # FIX: Gunakan standard DataLoader — UniqueRecordingBatchSampler
    # terlalu membatasi jumlah gradient updates per epoch
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if device.type == 'cuda' else False
    )

    # --- 4. Inisialisasi Model ---
    # ✅ GUNAKAN Config untuk parameter model
    model = MambaDrowsinessDetector(
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.get_output_dim(),
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
    
    if Config.is_classification():
        class_weights = compute_inverse_weight([sample['class_idx'] for sample in train_dataset.samples], num_classes=Config.NUM_CLASSES)
        criterion = WeightedCrossEntropyLoss(weight=class_weights.to(device))
    else:
        criterion = MAELoss()
    
    # TAMBAHKAN Learning Rate Scheduler
    scheduler = None
    use_warmup = getattr(Config, 'USE_WARMUP', False)
    warmup_epochs = getattr(Config, 'WARMUP_EPOCHS', 0)
    warmup_start_factor = getattr(Config, 'WARMUP_START_FACTOR', 0.1)

    if Config.USE_SCHEDULER:
        from torch.optim.lr_scheduler import ReduceLROnPlateau, LinearLR

        plateau_scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',  # Monitor val_loss (semakin kecil semakin baik)
            factor=Config.SCHEDULER_FACTOR,
            patience=Config.SCHEDULER_PATIENCE,
            verbose=True
        )

        if use_warmup and warmup_epochs > 0:
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,
                total_iters=warmup_epochs
            )
            scheduler = {
                'warmup': warmup_scheduler,
                'plateau': plateau_scheduler
            }
            print(f"✅ Warmup enabled: {warmup_epochs} epochs, start_factor={warmup_start_factor}")
        else:
            scheduler = {
                'warmup': None,
                'plateau': plateau_scheduler
            }

        print(f"✅ Learning Rate Scheduler enabled (ReduceLROnPlateau, patience={Config.SCHEDULER_PATIENCE})")

    # --- 7. Loop Pelatihan ---
    if Config.is_classification():
        best_val_acc = 0.0
        best_val_f1 = 0.0
        best_val_mae = None
        best_val_rmse = None
    else:
        best_val_mae = float('inf')
        best_val_rmse = float('inf')
        best_val_acc = None
        best_val_f1 = None
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
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            # Gradient clipping: cegah exploding gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            if Config.is_classification():
                train_preds_list.append(torch.argmax(outputs, dim=-1).detach().cpu())
            else:
                train_preds_list.append(outputs.detach().cpu())
            train_targets_list.append(labels.detach().cpu())
            
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
                outputs = model(signals)
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()
                
                if Config.is_classification():
                    val_preds_list.append(torch.argmax(outputs, dim=-1).detach().cpu())
                else:
                    val_preds_list.append(outputs.detach().cpu())
                val_targets_list.append(labels.detach().cpu())
                
                pbar_val.set_postfix({"loss": f"{val_loss.item():.4f}"})

        # ========================================
        # PERHITUNGAN METRIK
        # ========================================
        train_preds = torch.cat(train_preds_list)
        train_targets = torch.cat(train_targets_list)
        val_preds = torch.cat(val_preds_list)
        val_targets = torch.cat(val_targets_list)
        
        if Config.is_classification():
            train_acc = (train_preds == train_targets).float().mean().item()
            val_acc = (val_preds == val_targets).float().mean().item()

            # Compute F1 manually for binary or multiclass
            def compute_f1(preds, targets, num_classes):
                f1_scores = []
                for cls in range(num_classes):
                    tp = ((preds == cls) & (targets == cls)).sum().item()
                    fp = ((preds == cls) & (targets != cls)).sum().item()
                    fn = ((preds != cls) & (targets == cls)).sum().item()
                    if tp == 0:
                        f1_scores.append(0.0)
                        continue
                    precision = tp / (tp + fp + 1e-12)
                    recall = tp / (tp + fn + 1e-12)
                    f1_scores.append(2 * precision * recall / (precision + recall + 1e-12))
                return sum(f1_scores) / len(f1_scores)

            train_f1 = compute_f1(train_preds, train_targets, Config.NUM_CLASSES)
            val_f1 = compute_f1(val_preds, val_targets, Config.NUM_CLASSES)

            train_mae, train_rmse = 0.0, 0.0
            val_mae, val_rmse = 0.0, 0.0
            train_preds_binary, train_targets_binary = train_preds, train_targets
            val_preds_binary, val_targets_binary = val_preds, val_targets
        else:
            train_preds = train_preds.squeeze(-1)
            train_targets = train_targets.squeeze(-1)
            val_preds = val_preds.squeeze(-1)
            val_targets = val_targets.squeeze(-1)

            train_mae, train_rmse = compute_regression_metrics(train_preds, train_targets)
            val_mae, val_rmse = compute_regression_metrics(val_preds, val_targets)

            train_acc, train_preds_binary, train_targets_binary = get_classification_stats(train_preds, train_targets)
            val_acc, val_preds_binary, val_targets_binary = get_classification_stats(val_preds, val_targets)

            # Binary F1 from thresholded regression output
            def compute_binary_f1(preds, targets):
                tp = ((preds == 1) & (targets == 1)).sum().item()
                fp = ((preds == 1) & (targets == 0)).sum().item()
                fn = ((preds == 0) & (targets == 1)).sum().item()
                if tp == 0:
                    return 0.0
                precision = tp / (tp + fp + 1e-12)
                recall = tp / (tp + fn + 1e-12)
                return 2 * precision * recall / (precision + recall + 1e-12)

            train_f1 = compute_binary_f1(train_preds_binary, train_targets_binary)
            val_f1 = compute_binary_f1(val_preds_binary, val_targets_binary)

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        # Learning Rate Scheduler Step (warmup + plateau)
        if scheduler is not None:
            if scheduler['warmup'] is not None and epoch < warmup_epochs:
                scheduler['warmup'].step()

            scheduler['plateau'].step(avg_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = Config.LEARNING_RATE

        # ========================================
        # LOGGING KE WANDB
        # ========================================
        wandb_log = {
            "epoch": epoch + 1,
            "learning_rate": current_lr,
            "train/loss": avg_train_loss,
            "train/mae": train_mae,
            "train/rmse": train_rmse,
            "train/accuracy_threshold": train_acc,
            "train/f1_threshold": train_f1,
            "val/loss": avg_val_loss,
            "val/mae": val_mae,
            "val/rmse": val_rmse,
            "val/accuracy_threshold": val_acc,
            "val/f1_threshold": val_f1,
        }

        if Config.USE_WANDB:
            wandb.log(wandb_log)

        # Print hasil epoch
        print(f"\n📊 Epoch {epoch+1}/{Config.EPOCHS} Summary:")
        print(f"   Train - Loss: {avg_train_loss:.4f} | MAE: {train_mae:.4f} | RMSE: {train_rmse:.4f} | Acc(threshold): {train_acc:.4f} | F1(threshold): {train_f1:.4f}")
        print(f"   Val   - Loss: {avg_val_loss:.4f} | MAE: {val_mae:.4f} | RMSE: {val_rmse:.4f} | Acc(threshold): {val_acc:.4f} | F1(threshold): {val_f1:.4f}")
        print(f"   LR: {current_lr:.2e}")

        # ========================================
        # CONFUSION MATRIX (EVALUASI THRESHOLD)
        # ========================================
        if (epoch + 1) % cm_save_interval == 0 or epoch == Config.EPOCHS - 1:
            print(f"\n📈 Generating Confusion Matrix...")

            if Config.is_classification():
                if Config.NUM_CLASSES == 2:
                    class_names = ['Alert', 'Drowsy']
                    labels = [0, 1]
                else:
                    class_names = ['Alert', 'Low Vigilance', 'Drowsy']
                    labels = [0, 1, 2]
            else:
                class_names = ['Alert', 'Drowsy']
                labels = [0, 1]

            train_cm = plot_confusion_matrix(
                train_targets_binary, train_preds_binary,
                epoch=epoch+1, fold=current_fold,
                phase='train', save_dir=cm_save_dir,
                class_names=class_names, labels=labels
            )
            
            val_cm = plot_confusion_matrix(
                val_targets_binary, val_preds_binary,
                epoch=epoch+1, fold=current_fold,
                phase='val', save_dir=cm_save_dir,
                class_names=class_names, labels=labels
            )
            
            print(f"\n📋 Validation Set Classification Report:")
            val_class_metrics = print_classification_report(val_cm, class_names=class_names)

        # ========================================
        # SIMPAN MODEL TERBAIK & EARLY STOPPING
        # ========================================
        if Config.is_classification():
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_f1 = val_f1
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
                        'val_acc': val_acc,
                        'val_f1': val_f1,
                        'config': clean_config
                    }
                    torch.save(checkpoint, model_name)
                    if Config.USE_WANDB:
                        wandb.save(model_name)
                    
                    print(f"\n✅ Model terbaik disimpan: {model_name}")
                    print(f"   └─ Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}\n")
            else:
                patience_counter += 1  # ✅ Increment counter
        else:
            if val_mae < best_val_mae:
                best_val_mae = val_mae
                best_val_rmse = val_rmse
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
                        'val_mae': val_mae,
                        'val_rmse': val_rmse,
                        'config': clean_config
                    }
                    
                    torch.save(checkpoint, model_name)
                    if Config.USE_WANDB:
                        wandb.save(model_name)
                    
                    print(f"\n✅ Model terbaik disimpan: {model_name}")
                    print(f"   └─ Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f}\n")
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
    if Config.is_classification():
        print(f"✅ Best Validation Acc: {best_val_acc:.4f}")
        print(f"✅ Best Validation F1: {best_val_f1:.4f}")
    else:
        print(f"✅ Best Validation MAE: {best_val_mae:.4f}")
        print(f"✅ Best Validation RMSE: {best_val_rmse:.4f}")
    print("="*60 + "\n")
    
    if Config.USE_WANDB:
        if Config.is_classification():
            wandb.run.summary["best_val_acc"] = best_val_acc
            wandb.run.summary["best_val_f1"] = best_val_f1
        else:
            wandb.run.summary["best_val_mae"] = best_val_mae
            wandb.run.summary["best_val_rmse"] = best_val_rmse
    
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
            outputs = model(signals)
            final_val_preds.append(outputs.detach().cpu())
            final_val_targets.append(labels.cpu())
    
    final_val_preds = torch.cat(final_val_preds)
    final_val_targets = torch.cat(final_val_targets)

    if Config.is_classification():
        if final_val_preds.dim() > 1:
            final_val_preds = torch.argmax(final_val_preds, dim=-1)
        final_val_targets = final_val_targets.squeeze(-1) if final_val_targets.dim() > 1 else final_val_targets

        final_acc = (final_val_preds == final_val_targets).float().mean().item()
        final_f1 = 0.0
        for cls in range(Config.NUM_CLASSES):
            tp = ((final_val_preds == cls) & (final_val_targets == cls)).sum().item()
            fp = ((final_val_preds == cls) & (final_val_targets != cls)).sum().item()
            fn = ((final_val_preds != cls) & (final_val_targets == cls)).sum().item()
            if tp == 0:
                continue
            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            final_f1 += 2 * precision * recall / (precision + recall + 1e-12)
        final_f1 = final_f1 / Config.NUM_CLASSES

        if Config.NUM_CLASSES == 2:
            class_names = ['Alert', 'Drowsy']
            labels = [0, 1]
        else:
            class_names = ['Alert', 'Low Vigilance', 'Drowsy']
            labels = [0, 1, 2]

        final_cm = plot_confusion_matrix(
            final_val_targets, final_val_preds,
            epoch='final', fold=current_fold,
            phase='val_best_model', save_dir=cm_save_dir,
            class_names=class_names, labels=labels
        )
        
        print(f"\n📊 Final Classification Metrics: Acc={final_acc:.4f}, F1={final_f1:.4f}")
        print("\n📋 Final Model Classification Report:")
        print_classification_report(final_cm, class_names=class_names)
    else:
        final_val_preds = final_val_preds.squeeze(-1)
        final_val_targets = final_val_targets.squeeze(-1)

        final_mae, final_rmse = compute_regression_metrics(final_val_preds, final_val_targets)
        final_acc, final_preds_binary, final_targets_binary = get_classification_stats(final_val_preds, final_val_targets)
        
        final_cm = plot_confusion_matrix(
            final_targets_binary, final_preds_binary,
            epoch='final', fold=current_fold,
            phase='val_best_model', save_dir=cm_save_dir,
            class_names=['Alert', 'Drowsy'], labels=[0, 1]
        )
        
        print(f"\n📊 Final Regression Metrics: MAE={final_mae:.4f}, RMSE={final_rmse:.4f}, Accuracy(threshold)={final_acc:.4f}")
        print("\n📋 Final Model Classification Report:")
        print_classification_report(final_cm, class_names=['Alert', 'Drowsy'])

    if Config.USE_WANDB:
        wandb.finish()
    
    if Config.is_classification():
        return {
            'best_val_acc': best_val_acc,
            'best_val_f1': best_val_f1
        }
    else:
        return {
            'best_val_mae': best_val_mae,
            'best_val_rmse': best_val_rmse
        }

if __name__ == "__main__":
    # Train single fold
    # train(fold=0)
    
    # Fll 5-fold CV
    results = []
    for fold in range(Config.N_SPLITS):
        print(f"\n{'='*70}")
        print(f"📂 STARTING FOLD {fold + 1}/{Config.N_SPLITS}")
        print(f"{'='*70}\n")
        
        result = train(fold=fold)
        result['fold'] = fold
        results.append(result)
        
        if Config.is_classification():
            print(f"\n✅ Fold {fold + 1} completed: Acc={result['best_val_acc']:.4f}, F1={result['best_val_f1']:.4f}\n")
        else:
            print(f"\n✅ Fold {fold + 1} completed: MAE={result['best_val_mae']:.4f}, RMSE={result['best_val_rmse']:.4f}\n")
    
    # Ringkasan hasil
    print("\n" + "="*70)
    print("📊 5-FOLD CROSS VALIDATION RESULTS")
    print("="*70)
    if Config.is_classification():
        for r in results:
            print(f"Fold {r['fold']+1}: Acc={r['best_val_acc']:.4f}, F1={r['best_val_f1']:.4f}")
        mean_acc = np.mean([r['best_val_acc'] for r in results])
        std_acc = np.std([r['best_val_acc'] for r in results])
        mean_f1 = np.mean([r['best_val_f1'] for r in results])
        std_f1 = np.std([r['best_val_f1'] for r in results])
        print("-"*70)
        print(f"Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"Mean F1-Score: {mean_f1:.4f} ± {std_f1:.4f}")
    else:
        for r in results:
            print(f"Fold {r['fold']+1}: MAE={r['best_val_mae']:.4f}, RMSE={r['best_val_rmse']:.4f}")
        mean_mae = np.mean([r['best_val_mae'] for r in results])
        std_mae = np.std([r['best_val_mae'] for r in results])
        mean_rmse = np.mean([r['best_val_rmse'] for r in results])
        std_rmse = np.std([r['best_val_rmse'] for r in results])
        print("-"*70)
        print(f"Mean MAE: {mean_mae:.4f} ± {std_mae:.4f}")
        print(f"Mean RMSE: {mean_rmse:.4f} ± {std_rmse:.4f}")
    print("="*70)
