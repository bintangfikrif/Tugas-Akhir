import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter
from config import Config

class WeightedCrossEntropyLoss(nn.Module):
    """
    Fungsi Loss utama sesuai Proposal Tugas Akhir.
    Menggunakan bobot untuk menangani ketidakseimbangan kelas.
    """
    def __init__(self, weight=None):
        super().__init__()
        self.weight = weight
        
    def forward(self, logits, targets):
        return F.cross_entropy(logits, targets, weight=self.weight)
    
class MAELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss() # MAE Loss
        
    def forward(self, predictions, targets):
        # Pastikan tipe datanya float32 untuk regresi
        return self.criterion(predictions, targets)

def compute_inverse_weight(labels, num_classes=None):
    """
    Menghitung bobot menggunakan metode Inverse Class Frequency.
    Bobot = Total_Sampel / (Jumlah_Kelas * Sampel_Per_Kelas)
    """
    if num_classes is None:
        num_classes = Config.NUM_CLASSES
    counts = Counter(labels)
    total_samples = sum(counts.values())
    
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1) 
        weight = total_samples / (num_classes * count)
        weights.append(weight)
        
    return torch.tensor(weights, dtype=torch.float32)

def compute_regression_metrics(predictions, targets):
    # Balikkan ke skala KSS asli (1-9) sebelum dihitung
    preds_kss = predictions * Config.KSS_MAX
    targets_kss = targets * Config.KSS_MAX
    
    # Hitung MAE (Mean Absolute Error)
    mae = torch.mean(torch.abs(preds_kss - targets_kss))
    
    # Hitung RMSE (Root Mean Squared Error)
    mse = torch.mean((preds_kss - targets_kss)**2)
    rmse = torch.sqrt(mse)
    
    return mae.item(), rmse.item()

def get_classification_stats(predictions, targets, threshold=5.5):
    """
    Mengubah hasil regresi kembali ke biner untuk hitung akurasi/confusion matrix.
    Threshold 5.5: <= 5.5 Alert (0), > 5.5 Drowsy (1)
    """
    # Balikkan ke skala 1-9
    preds_kss = predictions * Config.KSS_MAX
    targets_kss = targets * Config.KSS_MAX
    
    # Thresholding menjadi biner (0 atau 1)
    preds_binary = (preds_kss > threshold).long()
    targets_binary = (targets_kss > threshold).long()
    
    # Hitung akurasi sederhana
    correct = (preds_binary == targets_binary).sum().item()
    total = targets_binary.size(0)
    acc = correct / total
    
    return acc, preds_binary, targets_binary