import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops import rearrange

class MambaBlock(nn.Module):
    """
    Blok Mamba tunggal dengan koneksi residual dan normalisasi.
    """
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super(MambaBlock, self).__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, L, D) -> Batch, Length, Dimension
        """
        # Pre-normalization dengan residual connection [cite: 352]
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return x + residual

class MambaDrowsinessDetector(nn.Module):
    """
    Model Deteksi Kantuk berbasis Mamba sesuai Proposal Tugas Akhir Bintang.
    - Input: 7 Channel (5 EEG, 2 EOG) [cite: 478]
    - Arsitektur: Conv1D -> Mamba Encoder (4 Block) -> Global Avg Pool -> Linear [cite: 543, 547]
    - Output: 3 Kelas (Alert, Low Vigilance, Drowsy) 
    """
    def __init__(
        self,
        in_channels=7,
        num_classes=3, 
        d_model=128,
        n_layers=4,    # Sesuai proposal: 4 blok Mamba [cite: 547]
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ):
        super(MambaDrowsinessDetector, self).__init__()
        
        # 1. Proyeksi Input: Menggunakan Conv1D untuk ekstraksi fitur temporal lokal 
        self.input_projection = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        
        # 2. Learnable Positional Embedding 
        # Max seq len untuk 30 detik pada 512Hz adalah 15360
        self.max_seq_len = 16000 
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.max_seq_len, d_model) * 0.02
        )
        
        # 3. Mamba Encoder: Stack dari 4 blok Mamba [cite: 547]
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # 4. Global Temporal Pooling [cite: 552]
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # 5. Classifier Head [cite: 552]
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes) # Output 3 Logits [cite: 553]
        )

    def forward(self, x):
        """
        Forward pass model.
        Input x: (B, 7, 15360) -> (Batch, Channels, Time) [cite: 544]
        """
        # Ekstraksi fitur lokal awal
        x = self.input_projection(x)  # (B, d_model, T)
        
        # Ubah dimensi untuk Mamba: (B, T, D)
        x = rearrange(x, 'b d t -> b t d')
        
        # Tambahkan positional encoding 
        T = x.shape[1]
        x = x + self.pos_encoding[:, :T, :]
        
        # Proses melalui 4 blok Mamba [cite: 547]
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        
        # Pooling untuk meringkas sekuens menjadi vektor representasi [cite: 551]
        x = rearrange(x, 'b t d -> b d t')
        x = self.pooling(x).squeeze(-1) # (B, D)
        
        # Klasifikasi akhir
        logits = self.classifier(x) # (B, 3)
        return logits

    def get_num_params(self):
        """Menghitung total parameter model [cite: 416]"""
        return sum(p.numel() for p in self.parameters())