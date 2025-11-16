"""
Mamba Model for Drowsiness Detection from EEG/EOG Signals

Architecture:
- Input: (B, C, T) where C=7 channels, T=512 time steps (1 second at 512 Hz)
- Channel projection: Conv1d to project 7 channels to d_model
- Mamba blocks: N layers of bidirectional Mamba SSM
- Output head: (B, 3) for 3-class classification
    - Class 0: Alert (KSS 1-3)
    - Class 1: Low Vigilance (KSS 4-6)
    - Class 2: Drowsy (KSS 7-9)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from einops import rearrange
import math

class MambaBlock(nn.Module):
    """
    Single Mamba block with residual connection and normalization.
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
            x: (B, L, D) where L is sequence length, D is d_model
        Returns:
            output: (B, L, D)
        """
        # Residual connection with pre-normalization
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = x + residual
        return x


class MambaEncoder(nn.Module):
    """
    Stack of Mamba blocks.
    """
    
    def __init__(self, n_layers, d_model, d_state=16, d_conv=4, expand=2):
        super(MambaEncoder, self).__init__()
        
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x):
        """
        Args:
            x: (B, L, D)
        Returns:
            output: (B, L, D)
        """
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return x


class MambaDrowsinessDetector(nn.Module):
    """
    Mamba-based Drowsiness Detection Model for 3-class classification.
    
    Classes:
        0: Alert (KSS 1-3)
        1: Low Vigilance (KSS 4-6)
        2: Drowsy (KSS 7-9)
    
    Args:
        in_channels: Number of input EEG/EOG channels (default: 7)
        num_classes: Number of classes (default: 3)
        d_model: Hidden dimension for Mamba (default: 128)
        n_layers: Number of Mamba layers (default: 4)
        d_state: SSM state expansion factor (default: 16)
        d_conv: Local convolution width (default: 4)
        expand: Block expansion factor (default: 2)
        dropout: Dropout rate (default: 0.1)
    """
    
    def __init__(
        self,
        in_channels=7,
        num_classes=3,  # CHANGED: 9 → 3
        d_model=128,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1,
    ):
        super(MambaDrowsinessDetector, self).__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Input projection: (B, C, T) -> (B, D, T) -> (B, T, D)
        self.input_projection = nn.Sequential(
            nn.Conv1d(in_channels, d_model, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        
        # Positional encoding (learnable)
        self.max_seq_len = 4096  # Maximum sequence length
        self.pos_encoding = nn.Parameter(
            torch.randn(1, self.max_seq_len, d_model) * 0.02
        )
        
        # Mamba encoder
        self.encoder = MambaEncoder(
            n_layers=n_layers,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Pooling
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Classification head for 3-class classification
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)  # CHANGED: Output 3 logits
        )
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T) input EEG/EOG signals
                B = batch size
                C = 7 channels (Fz, Cz, C3, C4, Pz, EOG-V, EOG-H)
                T = 512 time steps (1 second at 512 Hz)
            
        Returns:
            logits: (B, 3) logits for 3-class classification
        """
        B, C, T = x.shape
        
        # Input projection: (B, C, T) -> (B, D, T)
        x = self.input_projection(x)  # (B, d_model, T)
        
        # Rearrange: (B, D, T) -> (B, T, D)
        x = rearrange(x, 'b d t -> b t d')
        
        # Add positional encoding
        if T <= self.max_seq_len:
            x = x + self.pos_encoding[:, :T, :]
        else:
            # Interpolate positional encoding if sequence is longer
            pos_enc = F.interpolate(
                self.pos_encoding.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
            x = x + pos_enc
        
        # Mamba encoder: (B, T, D) -> (B, T, D)
        x = self.encoder(x)
        
        # Temporal pooling: (B, T, D) -> (B, D, 1) -> (B, D)
        x = rearrange(x, 'b t d -> b d t')
        x = self.pooling(x).squeeze(-1)  # (B, D)
        
        # Dropout
        x = self.dropout(x)
        
        # Classification head: (B, D) -> (B, 3)
        logits = self.classifier(x)
        
        return logits
    
    def get_num_params(self):
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ==================== Baseline: ResNet-18 1D ====================

class BasicBlock1D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    """ResNet-18 for 1D EEG signals with ordinal regression."""
    
    def __init__(self, block, layers, in_channels=7, num_classes=9, zero_init_residual=False):
        super(ResNet1D, self).__init__()
        self.inplanes = 64
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.5)
        
        # Classification head for 3-class classification
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock1D):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, return_class_logits=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        
        logits = self.classifier(x)
        return logits
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())
    
    def get_num_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def resnet18_1d(in_channels=7, num_classes=3):
    """Constructs a ResNet-18 model for 1D signals."""
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], in_channels=in_channels, num_classes=num_classes)


# ==================== Model Factory ====================

def create_model(model_name, in_channels=7, num_classes=3, **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_name: 'mamba' or 'resnet18'
        in_channels: Number of input channels
        num_classes: Number of output classes
        **kwargs: Additional model-specific arguments
    """
    if model_name.lower() == 'mamba':
        model = MambaDrowsinessDetector(
            in_channels=in_channels,
            num_classes=num_classes,
            d_model=kwargs.get('d_model', 128),
            n_layers=kwargs.get('n_layers', 4),
            d_state=kwargs.get('d_state', 16),
            d_conv=kwargs.get('d_conv', 4),
            expand=kwargs.get('expand', 2),
            dropout=kwargs.get('dropout', 0.1),
        )
    elif model_name.lower() == 'resnet18':
        model = resnet18_1d(in_channels=in_channels, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    return model


if __name__ == '__main__':
    print("="*80)
    print("Testing Mamba Drowsiness Detector")
    print("="*80)
    
    # Test Mamba model
    model = MambaDrowsinessDetector(
        in_channels=7,
        num_classes=9,
        d_model=128,
        n_layers=4,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.1
    )
    
    # Test input
    x = torch.randn(2, 7, 2560)  # (batch=2, channels=7, time=2560)
    
    print(f"Input shape: {x.shape}")
    print(f"Total parameters: {model.get_num_params():,}")
    print(f"Trainable parameters: {model.get_num_trainable_params():,}")
    
    # Forward pass
    ordinal_logits = model(x)
    print(f"\nOrdinal logits shape: {ordinal_logits.shape}")  # (2, 8)
    
    ordinal_logits, class_logits = model(x, return_class_logits=True)
    print(f"Class logits shape: {class_logits.shape}")  # (2, 9)
    
    print("\n" + "="*80)
    print("Testing ResNet-18 1D Baseline")
    print("="*80)
    
    # Test ResNet baseline
    model_resnet = resnet18_1d(in_channels=7, num_classes=9)
    
    print(f"Input shape: {x.shape}")
    print(f"Total parameters: {model_resnet.get_num_params():,}")
    print(f"Trainable parameters: {model_resnet.get_num_trainable_params():,}")
    
    ordinal_logits = model_resnet(x)
    print(f"\nOrdinal logits shape: {ordinal_logits.shape}")  # (2, 8)
    
    ordinal_logits, class_logits = model_resnet(x, return_class_logits=True)
    print(f"Class logits shape: {class_logits.shape}")  # (2, 9)