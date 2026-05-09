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
        count = counts.get(i, 1) # Hindari pembagian dengan nol
        weight = total_samples / (num_classes * count)
        weights.append(weight)
        
    return torch.tensor(weights, dtype=torch.float32)