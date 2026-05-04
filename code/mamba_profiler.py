import torch
import torch.nn as nn
from einops import rearrange

import warnings
warnings.filterwarnings("ignore")

class Config:
    IN_CHANNELS   = 7        # Jumlah channel EEG/PSG input
    OUTPUT_DIM    = 2        # Alert / Low Vigilant / Drowsy
    MAMBA_D_MODEL = 32       # Dimensi model Mamba
    MAMBA_N_LAYERS = 4       # Jumlah blok Mamba
    MAMBA_D_STATE = 16       # State dimension SSM
    MAMBA_D_CONV  = 4        # Kernel conv internal Mamba
    MAMBA_EXPAND  = 2        # Expansion factor


# ==============================================================
# MODEL DEFINITION
# ==============================================================

try:
    from mamba_ssm import Mamba

    class MambaBlock(nn.Module):
        def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.norm  = nn.LayerNorm(d_model)
            self.mamba = Mamba(d_model=d_model, d_state=d_state,
                               d_conv=d_conv, expand=expand)

        def forward(self, x):
            residual = x
            x = self.norm(x)
            x = self.mamba(x)
            return x + residual

    class MambaDrowsinessDetector(nn.Module):
        def __init__(
            self,
            in_channels = Config.IN_CHANNELS,
            num_classes = Config.OUTPUT_DIM,
            d_model     = Config.MAMBA_D_MODEL,
            n_layers    = Config.MAMBA_N_LAYERS,
            d_state     = Config.MAMBA_D_STATE,
            d_conv      = Config.MAMBA_D_CONV,
            expand      = Config.MAMBA_EXPAND,
            dropout     = 0.5,
        ):
            super().__init__()
            half_d = d_model // 2

            self.input_projection = nn.Sequential(
                nn.Conv1d(in_channels, half_d, kernel_size=16, stride=8,  padding=4, bias=False),
                nn.BatchNorm1d(half_d),
                nn.GELU(),
                nn.Conv1d(half_d, d_model,    kernel_size=8,  stride=4,  padding=2, bias=False),
                nn.BatchNorm1d(d_model),
                nn.GELU(),
            )

            self.max_seq_len  = 512
            self.pos_encoding = nn.Parameter(torch.randn(1, self.max_seq_len, d_model) * 0.02)

            self.layers = nn.ModuleList([
                MambaBlock(d_model, d_state, d_conv, expand)
                for _ in range(n_layers)
            ])

            self.final_norm = nn.LayerNorm(d_model)
            self.pooling    = nn.AdaptiveAvgPool1d(1)

            self.classifier = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model // 2, num_classes),
            )

        def forward(self, x):
            x = self.input_projection(x)
            x = rearrange(x, 'b d t -> b t d')
            T = x.shape[1]
            x = x + self.pos_encoding[:, :T, :]
            for layer in self.layers:
                x = layer(x)
            x = self.final_norm(x)
            x = rearrange(x, 'b t d -> b d t')
            x = self.pooling(x).squeeze(-1)
            return self.classifier(x)

        def get_num_params(self):
            return sum(p.numel() for p in self.parameters())

    MAMBA_AVAILABLE = True

except ImportError:
    MAMBA_AVAILABLE = False
    print("mamba_ssm tidak terinstall")


# ==============================================================
# HELPER FUNCTIONS
# ==============================================================

def format_bytes(num_bytes):
    for unit in ['B', 'KB', 'MB', 'GB']:
        if num_bytes < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.2f} TB"


def count_parameters(model):
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen    = total - trainable
    return total, trainable, frozen


def get_per_component_params(model):
    """Breakdown parameter per komponen utama model."""
    components = {
        "input_projection (Conv1d ×2 + BN)": model.input_projection,
        "pos_encoding (learnable)":           None,   # parameter langsung
        "Mamba layers (4× MambaBlock)":       model.layers,
        "final_norm (LayerNorm)":             model.final_norm,
        "classifier head (Linear ×2)":        model.classifier,
    }
    results = []
    for name, module in components.items():
        if module is None:
            # pos_encoding adalah nn.Parameter, bukan modul
            params = model.pos_encoding.numel()
        else:
            params = sum(p.numel() for p in module.parameters())
        results.append((name, params))
    return results


def estimate_file_size(model):
    """Estimasi ukuran state_dict .pth untuk berbagai dtype."""
    total_params  = sum(p.numel() for p in model.parameters())
    total_buffers = sum(b.numel() for b in model.buffers())   # BatchNorm running stats, dll.
    total_elems   = total_params + total_buffers

    fp32 = total_elems * 4
    fp16 = total_elems * 2
    int8 = total_elems * 1
    return fp32, fp16, int8


def estimate_activation_memory(model, input_shape, device):
    """Ukur memori aktivasi via forward hooks (inference, batch=1)."""
    activation_bytes = []
    hooks = []

    def hook_fn(module, inp, output):
        if isinstance(output, torch.Tensor):
            activation_bytes.append(output.numel() * output.element_size())
        elif isinstance(output, (tuple, list)):
            for o in output:
                if isinstance(o, torch.Tensor):
                    activation_bytes.append(o.numel() * o.element_size())

    for module in model.modules():
        if len(list(module.children())) == 0:
            hooks.append(module.register_forward_hook(hook_fn))

    model.eval()
    with torch.no_grad():
        dummy = torch.randn(*input_shape).to(device)
        _ = model(dummy)

    for h in hooks:
        h.remove()

    return sum(activation_bytes)


# ==============================================================
# KOMPLEKSITAS KOMPUTASI — Metode Hybrid (thop + rumus Mamba)
# ==============================================================

def evaluate_model_complexity(model, device, input_shape=(1, 7, 15360)):
    print("\n" + "="*60)
    print("📊 EVALUASI KOMPLEKSITAS KOMPUTASI")
    print("="*60)

    dummy_input = torch.randn(input_shape).to(device)
    model.eval()

    with torch.no_grad():
        # ── 1. thop: MACs untuk Conv1d, Linear, BN, LN, Pooling ──
        try:
            from thop import profile
            macs_thop, params_thop = profile(model, inputs=(dummy_input,), verbose=False)
        except Exception as e:
            print(f"  ⚠️  thop error: {e}")
            macs_thop  = 0
            params_thop = 0

        # ── 2. Total parameter dari model.get_num_params() ────────
        total_params = model.get_num_params()

        # ── 3. MACs inti Mamba — rumus analitik ───────────────────
        B = input_shape[0]              # batch size = 1
        L = 15360 // 8 // 4            # = 480
        D = Config.MAMBA_D_MODEL       # = 32
        N = Config.MAMBA_D_STATE       # = 16

        flops_mamba_per_layer  = 9 * B * L * D * N
        flops_mamba_total      = flops_mamba_per_layer * Config.MAMBA_N_LAYERS
        macs_mamba_core        = flops_mamba_total / 2

        # ── 4. Total MACs ──────────────────────────────────────────
        total_macs = macs_thop + macs_mamba_core

    # ── Format output ──────────────────────────────────────────
    gflops          = total_macs / 1e9
    params_million  = total_params / 1e6
    macs_str        = f"{total_macs / 1e6:.3f}M"
    params_str      = f"{total_params / 1e3:.3f}K"

    print(f"\n  Rincian Kalkulasi MACs:")
    print(f"  ├─ MACs dari thop (Conv1d/Linear/BN/LN)  : {macs_thop/1e6:.3f}M")
    print(f"  ├─ MACs Mamba core (rumus analitik)      : {macs_mamba_core/1e6:.3f}M")
    print(f"  │    └─ B={B}, L={L}, D={D}, N={N}, layers={Config.MAMBA_N_LAYERS}")
    print(f"  │       FLOPs/layer = 9×{B}×{L}×{D}×{N} = {flops_mamba_per_layer:,}")
    print(f"  │       Total FLOPs = {flops_mamba_total:,}")
    print(f"  │       MACs (÷2)   = {macs_mamba_core:,.0f}")
    print(f"  └─ Total MACs                            : {macs_str}")

    print(f"\n📈 Hasil Evaluasi:")
    print(f"   ├─ Jumlah Parameter : {params_str} ({params_million:.6f}M)")
    print(f"   ├─ MACs             : {macs_str}")
    print(f"   └─ GFLOPs           : {gflops:.6f}")

    return gflops, params_million, macs_str, params_str


# ==============================================================
# MAIN PROFILER
# ==============================================================

def run_full_profile():
    if not MAMBA_AVAILABLE:
        print("❌ Tidak bisa lanjut — mamba_ssm tidak terinstall.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = MambaDrowsinessDetector().to(device)

    INPUT_SHAPE = (1, Config.IN_CHANNELS, 15360)   # (B, C, L)

    print("\n" + "="*60)
    print("  MAMBA DROWSINESS DETECTOR — FULL PROFILER")
    print("="*60)
    print(f"  Device       : {device}")
    print(f"  Input shape  : {INPUT_SHAPE}  (B, C, L)")
    print(f"  Config       : d_model={Config.MAMBA_D_MODEL}, n_layers={Config.MAMBA_N_LAYERS},",
          f"d_state={Config.MAMBA_D_STATE}, expand={Config.MAMBA_EXPAND}")

    # ── 1. Parameter Count ─────────────────────────────────────
    total, trainable, frozen = count_parameters(model)
    print("\n📦 PARAMETER COUNT")
    print(f"  Total Parameters   : {total:>10,}  ({total/1e3:.3f}K  /  {total/1e6:.6f}M)")
    print(f"  Trainable          : {trainable:>10,}")
    print(f"  Frozen             : {frozen:>10,}")

    print("\n  Per-Component Breakdown:")
    layer_info = get_per_component_params(model)
    for name, params in layer_info:
        pct = params / total * 100
        bar = "█" * int(pct / 5)
        print(f"    {name:<45} {params:>8,}  ({pct:5.1f}%)  {bar}")

    # ── 2. Ukuran File .pth ────────────────────────────────────
    fp32, fp16, int8 = estimate_file_size(model)
    print("\n💾 ESTIMASI UKURAN FILE .pth (state_dict only)")
    print(f"  float32 (default)  : {format_bytes(fp32)}")
    print(f"  float16 (half)     : {format_bytes(fp16)}")
    print(f"  int8 (quantized)   : {format_bytes(int8)}")

    # ── 3. Activation Memory ───────────────────────────────────
    print("\n⚡ ACTIVATION MEMORY (inference, batch=1)")
    try:
        act_mem = estimate_activation_memory(model, INPUT_SHAPE, device)
        print(f"  Total activations  : {format_bytes(act_mem)}")
    except Exception as e:
        act_mem = 0
        print(f"  Gagal diukur via hooks: {e}")

    # ── 4. Runtime Memory ──────────────────────────────────────
    print("\n🖥️  ESTIMASI KEBUTUHAN MEMORI RUNTIME")
    print("  ── Inference Mode ──────────────────────────────────")
    print(f"     Weights (fp32)   : {format_bytes(fp32)}")
    if act_mem:
        print(f"     Activations     : {format_bytes(act_mem)}")
        print(f"     TOTAL estimasi  : {format_bytes(fp32 + act_mem)}")
    else:
        print(f"     Activations     : (tidak terukur)")
        print(f"     TOTAL estimasi  : {format_bytes(fp32)} + activations")

    print("")
    print("  ── Training Mode (Adam, fp32) ──────────────────────")
    grad_mem  = total * 4
    adam_mem  = total * 4 * 2
    train_mem = fp32 + (act_mem or 0) + grad_mem + adam_mem
    print(f"     Weights         : {format_bytes(fp32)}")
    print(f"     Activations     : {format_bytes(act_mem) if act_mem else 'N/A'}")
    print(f"     Gradients       : {format_bytes(grad_mem)}")
    print(f"     Adam states     : {format_bytes(adam_mem)}")
    print(f"     TOTAL estimasi  : {format_bytes(train_mem)}")

    # ── 5. GFLOPs via hybrid method ────────────────────────────
    gflops, params_m, macs_str, params_str = evaluate_model_complexity(
        model, device, input_shape=INPUT_SHAPE
    )

    # ── 6. Ringkasan Akhir ─────────────────────────────────────
    print("\n" + "="*60)
    print("  RINGKASAN AKHIR")
    print("="*60)
    print(f"  Parameter         : {params_str}  ({params_m:.6f}M)")
    print(f"  MACs              : {macs_str}")
    print(f"  GFLOPs            : {gflops:.6f}")
    print(f"  File .pth (fp32)  : {format_bytes(fp32)}")
    if act_mem:
        print(f"  VRAM inference    : ~{format_bytes(fp32 + act_mem)}")
    print(f"  VRAM training*    : ~{format_bytes(train_mem)}")
    print(f"  (* estimasi, batch=1, Adam fp32)")
    print("="*60 + "\n")

    return {
        "params_total":   total,
        "params_str":     params_str,
        "gflops":         gflops,
        "macs_str":       macs_str,
        "file_fp32_bytes": fp32,
        "file_fp16_bytes": fp16,
        "activation_bytes": act_mem,
    }


if __name__ == "__main__":
    run_full_profile()