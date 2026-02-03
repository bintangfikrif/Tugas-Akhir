import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops import rearrange
from config import Config

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
        # Pre-normalization dengan residual connection 
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return x + residual

class MambaDrowsinessDetector(nn.Module):
    def __init__(
        self,
        in_channels=Config.IN_CHANNELS,
        num_classes=Config.NUM_CLASSES, 
        d_model=Config.MAMBA_D_MODEL,
        n_layers=Config.MAMBA_N_LAYERS,    
        d_state=Config.MAMBA_D_STATE,
        d_conv=Config.MAMBA_D_CONV,
        expand=Config.MAMBA_EXPAND,
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
        
        # 3. Mamba Encoder: Stack dari 4 blok Mamba
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.LayerNorm(d_model)
        
        # 4. Global Temporal Pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # 5. Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes) # Output 3 Logits [cite: 553]
        )

    def forward(self, x):
        # Ekstraksi fitur lokal awal
        x = self.input_projection(x)  # (B, d_model, T)
        
        # Ubah dimensi untuk Mamba: (B, T, D)
        x = rearrange(x, 'b d t -> b t d')
        
        # Tambahkan positional encoding 
        T = x.shape[1]
        x = x + self.pos_encoding[:, :T, :]
        
        # Proses melalui 4 blok Mamba
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        
        # Pooling untuk meringkas sekuens menjadi vektor representasi
        x = rearrange(x, 'b t d -> b d t')
        x = self.pooling(x).squeeze(-1) # (B, D)
        
        # Klasifikasi akhir
        logits = self.classifier(x) # (B, 3)
        return logits

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())