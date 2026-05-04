import torch
import torch.nn as nn
import torchvision.models as models

import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# 1. DEFINISI MODEL
# ==============================================================

class MultimodalFeatureCoupledModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()

        # Image Encoder: ResNet18
        self.resnet = models.resnet18(weights=None)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 512)

        # Time-Series Encoder: LSTM
        self.lstm = nn.LSTM(input_size=9, hidden_size=128, num_layers=1, batch_first=True)
        self.lstm_fc = nn.Linear(128, 512)

        # Classifier
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, img, psg):
        img_features = self.resnet(img)

        lstm_out, (h_n, _) = self.lstm(psg)
        psg_features = self.lstm_fc(h_n[-1])

        # Min-Max Normalization
        img_norm = (img_features - img_features.min(1, keepdim=True)[0]) / \
                   (img_features.max(1, keepdim=True)[0] - img_features.min(1, keepdim=True)[0] + 1e-8)
        psg_norm = (psg_features - psg_features.min(1, keepdim=True)[0]) / \
                   (psg_features.max(1, keepdim=True)[0] - psg_features.min(1, keepdim=True)[0] + 1e-8)

        # Feature Coupling
        coupled = img_norm * psg_norm
        return self.classifier(coupled)


# ==============================================================
# 2. HELPER FUNCTIONS
# ==============================================================

def format_bytes(num_bytes):
    """Format bytes ke unit yang readable."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def count_parameters(model):
    """Hitung total parameter, trainable, dan non-trainable."""
    total       = sum(p.numel() for p in model.parameters())
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen      = total - trainable
    return total, trainable, frozen


def estimate_file_size(model):
    """
    Estimasi ukuran file .pth.
    torch.save() menyimpan state_dict (weights saja), bukan full model.
    Ada sedikit overhead dari pickle/metadata (~beberapa KB), diabaikan di sini.
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(b.numel() for b in model.buffers())  # e.g. BatchNorm running stats
    total_elements = total_params + total_buffers

    size_fp32 = total_elements * 4   # float32
    size_fp16 = total_elements * 2   # float16 / half precision
    size_int8 = total_elements * 1   # int8 quantized

    return size_fp32, size_fp16, size_int8


def estimate_activation_memory(model, img_shape, psg_shape, device, dtype=torch.float32):
    activation_sizes = []
    hooks = []

    def hook_fn(module, input, output):
        if isinstance(output, torch.Tensor):
            activation_sizes.append(output.numel() * output.element_size())
        elif isinstance(output, (tuple, list)):
            for o in output:
                if isinstance(o, torch.Tensor):
                    activation_sizes.append(o.numel() * o.element_size())

    # Pasang hook ke semua sub-modul
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # leaf modules only
            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        dummy_img = torch.randn(*img_shape, dtype=dtype).to(device)
        dummy_psg = torch.randn(*psg_shape, dtype=dtype).to(device)
        _ = model(dummy_img, dummy_psg)

    for h in hooks:
        h.remove()

    total_activation = sum(activation_sizes)
    return total_activation


def estimate_runtime_memory(model, img_shape, psg_shape):
    """
    Estimasi total kebutuhan RAM/VRAM saat runtime:
    - Inference mode  : weights + activations
    - Training mode   : weights + activations + gradients + optimizer states (Adam)
    """
    total_params, _, _ = count_parameters(model)
    size_fp32, _, _ = estimate_file_size(model)

    activation_mem = estimate_activation_memory(model, img_shape, psg_shape)

    # ── Inference ──────────────────────────────────────────
    inference_mem = size_fp32 + activation_mem

    # ── Training (Adam optimizer) ───────────────────────────
    gradient_mem  = total_params * 4
    adam_mem      = total_params * 4 * 2
    training_mem  = size_fp32 + activation_mem + gradient_mem + adam_mem

    return inference_mem, training_mem, activation_mem


def get_per_layer_params(model):
    """Breakdown parameter per komponen utama."""
    components = {
        "ResNet18 (image encoder)": model.resnet,
        "LSTM (time-series encoder)": model.lstm,
        "LSTM FC projection": model.lstm_fc,
        "Classifier (FC)": model.classifier,
    }
    results = []
    for name, module in components.items():
        params = sum(p.numel() for p in module.parameters())
        results.append((name, params))
    return results


# ==============================================================
# 3. MAIN — RUN PROFILING
# ==============================================================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MultimodalFeatureCoupledModel(num_classes=3).to(device)

    # Input shapes sesuai paper
    IMG_SHAPE = (1, 1, 75, 75)    # (batch=1, channel=1, H=75, W=75)
    PSG_SHAPE = (1, 512, 9)       # (batch=1, timesteps=512, channels=9)

    print("\n" + "="*60)
    print("  MODEL PROFILER — Cao et al. (2025) Replication")
    print("  Multimodal Feature Coupled Model")
    print("="*60)
    print(f"  Device: {device}")

    # ── Parameter Count ────────────────────────────────────
    total, trainable, frozen = count_parameters(model)
    print("\n📦 PARAMETER COUNT")
    print(f"  Total Parameters   : {total:>12,}  ({total/1e6:.4f} M)")
    print(f"  Trainable          : {trainable:>12,}")
    print(f"  Frozen (non-train) : {frozen:>12,}")

    # ── Per-layer breakdown ────────────────────────────────
    print("\n  Per-Component Breakdown:")
    layer_info = get_per_layer_params(model)
    for name, params in layer_info:
        pct = params / total * 100
        print(f"    {name:<35} {params:>10,}  ({pct:.1f}%)")

    # ── File Size Estimate ─────────────────────────────────
    fp32, fp16, int8 = estimate_file_size(model)
    print("\n💾 ESTIMASI UKURAN FILE .pth (state_dict)")
    print(f"  float32 (default)  : {format_bytes(fp32)}")
    print(f"  float16 (half)     : {format_bytes(fp16)}")
    print(f"  int8 (quantized)   : {format_bytes(int8)}")

    # ── Activation Memory ──────────────────────────────────
    activation_mem = estimate_activation_memory(model, IMG_SHAPE, PSG_SHAPE)
    print(f"\n⚡ ACTIVATION MEMORY (forward pass, batch=1)")
    print(f"  Total activations  : {format_bytes(activation_mem)}")

    # ── Runtime Memory ─────────────────────────────────────
    inference_mem, training_mem, _ = estimate_runtime_memory(model, IMG_SHAPE, PSG_SHAPE)
    print(f"\n🖥️  ESTIMASI KEBUTUHAN MEMORI RUNTIME")
    print(f"  ── Inference Mode ──────────────────────────────")
    print(f"     Weights (fp32)  : {format_bytes(fp32)}")
    print(f"     Activations     : {format_bytes(activation_mem)}")
    print(f"     TOTAL           : {format_bytes(inference_mem)}  ← kebutuhan VRAM/RAM minimum")
    print(f"")
    print(f"  ── Training Mode (Adam, fp32) ──────────────────")
    print(f"     Weights         : {format_bytes(fp32)}")
    print(f"     Activations     : {format_bytes(activation_mem)}")
    print(f"     Gradients       : {format_bytes(total * 4)}")
    print(f"     Adam states     : {format_bytes(total * 4 * 2)}")
    print(f"     TOTAL           : {format_bytes(training_mem)}")

    # ── GFLOPs via thop  ─────────────────────────
    print(f"\n📊 GFLOPs")
    try:
        from thop import profile, clever_format
        dummy_img = torch.randn(*IMG_SHAPE).to(device)
        dummy_psg = torch.randn(*PSG_SHAPE).to(device)
        macs, params_thop = profile(model, inputs=(dummy_img, dummy_psg), verbose=False)
        macs_str, params_str = clever_format([macs, params_thop], "%.4f")
        print(f"  MACs (thop)        : {macs_str}  →  {macs/1e9:.4f} GFLOPs")
        print(f"  Params (thop)      : {params_str}")
    except ImportError:
        print("  thop tidak terinstall")

if __name__ == "__main__":
    main()