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

def get_evaluation_metrics(predictions, targets):
    """
    Disesuaikan untuk mendukung output BCE (Binary) 
    maupun CrossEntropy (Categorical).
    """
    # Pastikan tipe data sama untuk perbandingan
    predictions = predictions.to(targets.device).float()
    targets = targets.float()

    # Accuracy: (Prediksi == Target)
    # Untuk BCE, predictions sudah berupa 0.0 atau 1.0 dari logic (logits > 0)
    acc = (predictions == targets).float().mean()
    
    metrics = {}
    # Loop sesuai jumlah kelas (2 untuk biner)
    for cls in range(Config.NUM_CLASSES):
        # Filter sampel yang termasuk kelas 'cls'
        tp = ((predictions == cls) & (targets == cls)).sum().float()
        fp = ((predictions == cls) & (targets != cls)).sum().float()
        fn = ((predictions != cls) & (targets == cls)).sum().float()
        
        precision = tp / (tp + fp + 1e-6) 
        recall = tp / (tp + fn + 1e-6)   
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6) 
        metrics[f'class_{cls}'] = {'p': precision, 'r': recall, 'f1': f1}
        
    return acc, metrics