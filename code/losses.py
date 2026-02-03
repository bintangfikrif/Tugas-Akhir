import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import Counter

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

def compute_inverse_weight(labels, num_classes=3):
    """
    Menghitung bobot menggunakan metode Inverse Class Frequency.
    Bobot = Total_Sampel / (Jumlah_Kelas * Sampel_Per_Kelas)
    """
    counts = Counter(labels)
    total_samples = sum(counts.values())
    
    weights = []
    for i in range(num_classes):
        count = counts.get(i, 1) # Hindari pembagian dengan nol
        weight = total_samples / (num_classes * count)
        weights.append(weight)
        
    return torch.tensor(weights, dtype=torch.float32)

def get_evaluation_metrics(predictions, targets):
    # Accuracy: (TP + TN) / Total
    acc = (predictions == targets).float().mean()
    
    # Per-class metrics menggunakan rata-rata makro
    # Cocok untuk data tidak seimbang 
    metrics = {}
    for cls in range(3):
        tp = ((predictions == cls) & (targets == cls)).sum().float()
        fp = ((predictions == cls) & (targets != cls)).sum().float()
        fn = ((predictions != cls) & (targets == cls)).sum().float()
        
        precision = tp / (tp + fp + 1e-6) 
        recall = tp / (tp + fn + 1e-6)   
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6) 
        metrics[f'class_{cls}'] = {'p': precision, 'r': recall, 'f1': f1}
        
    return acc, metrics