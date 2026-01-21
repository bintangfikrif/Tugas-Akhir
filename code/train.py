import os
import torch
import numpy as np
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

# Mengimpor modul kustom yang telah disesuaikan dengan proposal
from datareader import EEGDataset, collate_fn
from models import MambaDrowsinessDetector
from losses import WeightedCrossEntropyLoss, compute_inverse_weight, get_evaluation_metrics

def train():
    # --- 1. Konfigurasi Eksperimen (Sesuai Bab III) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    window_sec = 30       # Jendela 30 detik sesuai standar klinis AASM
    n_splits = 5          # 5-Fold Cross Validation
    batch_size = 16       # Batch size optimal untuk GPU T4
    epochs = 50           # Jumlah iterasi pelatihan
    lr = 1e-4             # Learning rate untuk optimizer AdamW
    current_fold = 0      # Indeks fold saat ini (0-4)
    
    # --- 2. Inisialisasi Weights & Biases (WandB) ---
    wandb.init(
        project="Drowsiness-Mamba-DROZY",
        name=f"Mamba_SubjectWise_Fold_{current_fold}",
        config={
            "architecture": "Mamba",
            "layers": 4,           # 4 blok Mamba Encoder
            "d_model": 128,
            "window_sec": window_sec,
            "n_splits": n_splits,
            "batch_size": batch_size,
            "learning_rate": lr,
            "dataset": "DROZY"     # Menggunakan dataset standar DROZY
        }
    )

    # --- 3. Persiapan Dataset & Dataloader (Subject-Wise) ---
    # Memastikan tidak ada kebocoran data antar subjek
    train_dataset = EEGDataset(
        data_dir='psg', 
        csv_path='label/labels.csv',
        fold=current_fold, 
        split='train', 
        n_splits=n_splits,
        window_sec=window_sec
    )
    
    val_dataset = EEGDataset(
        data_dir='psg', 
        csv_path='label/labels.csv',
        fold=current_fold, 
        split='val', 
        n_splits=n_splits,
        window_sec=window_sec
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # --- 4. Penanganan Ketidakseimbangan Data ---
    # Menghitung bobot kelas otomatis menggunakan Inverse Class Frequency
    train_labels = [item[1] for item in train_dataset.windows]
    class_weights = compute_inverse_weight(train_labels, num_classes=3).to(device)
    print(f"Bobot Kelas (Fold {current_fold}): {class_weights}")

    # --- 5. Inisialisasi Model, Loss, dan Optimizer ---
    model = MambaDrowsinessDetector(
        in_channels=7,            # 5 EEG + 2 EOG
        num_classes=3,            # Alert, Low Vigilance, Drowsy
        d_model=128, 
        n_layers=4
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = WeightedCrossEntropyLoss(weight=class_weights)

    # --- 6. Loop Pelatihan ---
    best_val_acc = 0
    for epoch in range(epochs):
        # Fase Training
        model.train()
        total_train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
        for signals, labels in pbar:
            signals, labels = signals.to(device), labels.to(device)
            
            optimizer.zero_grad()
            logits = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

        # Fase Validasi
        model.eval()
        all_preds, all_targets = [], []
        total_val_loss = 0
        with torch.no_grad():
            for signals, labels in val_loader:
                signals, labels = signals.to(device), labels.to(device)
                logits = model(signals)
                val_loss = criterion(logits, labels)
                total_val_loss += val_loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(labels.cpu())

        # Perhitungan Metrik Evaluasi
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        acc, metrics = get_evaluation_metrics(all_preds, all_targets)
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        # Logging ke WandB Dashboard
        wandb.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_acc": acc.item(),
            "f1_alert": metrics['class_0']['f1'],
            "f1_low_vigilance": metrics['class_1']['f1'],
            "f1_drowsy": metrics['class_2']['f1']
        })

        print(f"Epoch {epoch} | Val Acc: {acc:.4f} | Val Loss: {avg_val_loss:.4f}")

        # Simpan Model Terbaik Berdasarkan Akurasi Validasi
        if acc > best_val_acc:
            best_val_acc = acc
            model_name = f"best_mamba_fold{current_fold}.pt"
            torch.save(model.state_dict(), model_name)
            wandb.save(model_name) # Unggah ke Cloud WandB
            print(f"✅ Model terbaik disimpan: {model_name}")

    wandb.finish()

if __name__ == "__main__":
    train()