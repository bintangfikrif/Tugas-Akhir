import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
import mne
from config import Config
from datareader import EEGDataset, apply_bandpass, EEGAugmentation

def plot_signal(signal, channel_name, title, filename):
    """
    Plot satu channel sinyal dan simpan ke file.
    """
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # Time axis dalam detik
    time_axis = np.arange(len(signal)) / Config.SAMPLE_RATE
    
    ax.plot(time_axis, signal, linewidth=0.8, alpha=0.8)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Time (seconds)', fontsize=11)
    ax.set_ylabel('Amplitude (μV)', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add statistics ke plot
    mean_val = np.mean(signal)
    std_val = np.std(signal)
    min_val = np.min(signal)
    max_val = np.max(signal)
    
    stats_text = f'Mean: {mean_val:.2f} | Std: {std_val:.2f} | Min: {min_val:.2f} | Max: {max_val:.2f}'
    ax.text(0.5, -0.15, stats_text, transform=ax.transAxes, 
            ha='center', fontsize=10, style='italic', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Plot disimpan: {filename}")
    plt.close(fig)

def visualize_augmentation_samples():
    """
    Visualisasikan augmentasi dengan 3 plot terpisah:
    1. Original signal
    2. Signal + Gaussian noise
    3. Signal + Amplitude scaling
    """
    # Konfigurasi dataset
    dataset = EEGDataset(
        data_dir=Config.DATA_DIR,
        csv_path=os.path.join('label/labels.csv'),
        fold=0,
        split='train',
        n_splits=Config.N_SPLITS,
        window_sec=Config.WINDOW_SEC,
        stride_sec=Config.STRIDE_SEC,
        use_augmentation=False  # Matikan augmentasi di dataset, kita manual apply
    )

    if len(dataset) == 0:
        print("Dataset kosong!")
        return

    # Ambil sample pertama
    idx = 0
    info = dataset.samples[idx]
    file_path = info['file_path']
    start_sec = info['start_sec']

    print(f"\n📊 Visualisasi Augmentasi")
    print(f"   File: {os.path.basename(file_path)}")
    print(f"   Subject: {info['subject_id']}, KSS: {info['label']}")
    print(f"   Start: {start_sec:.1f}s\n")

    # Baca sinyal mentah
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose='error')
    raw.pick(['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H'])
    raw.crop(tmin=start_sec, tmax=start_sec + Config.WINDOW_SEC, include_tmax=False)

    # Downsampling jika perlu
    if getattr(Config, 'USE_DOWNSAMPLE', False):
        target_sr = Config.DOWNSAMPLE_RATE
        if target_sr != Config.ORIGINAL_SAMPLE_RATE:
            raw.resample(target_sr, npad='auto')

    current_sr = Config.DOWNSAMPLE_RATE if getattr(Config, 'USE_DOWNSAMPLE', False) else Config.ORIGINAL_SAMPLE_RATE
    signal_raw = raw.get_data() * 1e6  # Volt -> μV

    # Bandpass filter jika aktif
    if getattr(Config, 'USE_BANDPASS_FILTER', True):
        signal_raw = apply_bandpass(signal_raw, sfreq=current_sr)

    # Normalisasi sesuai mode
    norm_mode = getattr(Config, 'NORMALIZATION', 'window')
    if norm_mode == 'subject' and hasattr(dataset, 'subject_stats'):
        subject_id = info.get('subject_id', os.path.basename(file_path).split('-')[0])
        subject_stats = dataset.subject_stats.get(subject_id)
        if subject_stats is not None:
            mean, std = subject_stats
            signal_normalized = (signal_raw - mean) / (std + 1e-6)
        else:
            signal_normalized = (signal_raw - np.mean(signal_raw, axis=1, keepdims=True)) / (np.std(signal_raw, axis=1, keepdims=True) + 1e-6)
    else:
        signal_normalized = (signal_raw - np.mean(signal_raw, axis=1, keepdims=True)) / (np.std(signal_raw, axis=1, keepdims=True) + 1e-6)

    # Convert ke torch untuk augmentasi
    signal_tensor = torch.tensor(signal_normalized, dtype=torch.float32)

    # Pilih channel pertama (Fz) untuk visualisasi
    channel_idx = 0
    channels = ['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H']
    channel_name = channels[channel_idx]

    # 1. Original signal
    signal_original = signal_tensor[channel_idx].numpy()

    # 2. Signal + Gaussian noise
    aug = EEGAugmentation()
    signal_with_noise = signal_tensor.clone()
    # Set probability = 1.0 agar selalu diterapkan (untuk demo)
    np.random.seed(42)  # Untuk reproducibility
    noise = torch.randn_like(signal_with_noise) * Config.AUG_GAUSSIAN_NOISE_STD
    signal_with_noise = signal_with_noise + noise
    signal_with_noise_np = signal_with_noise[channel_idx].numpy()

    # 3. Signal + Amplitude scaling
    signal_with_scale = signal_tensor.clone()
    np.random.seed(42)
    scale = np.random.uniform(*Config.AUG_AMPLITUDE_SCALE_RANGE)
    signal_with_scale = signal_with_scale * scale
    signal_with_scale_np = signal_with_scale[channel_idx].numpy()

    # Create output directory
    os.makedirs('augmentation_visualizations', exist_ok=True)

    # Plot dan simpan 3 gambar terpisah
    title_original = f"Original Signal - {channel_name} Channel (Subject {info['subject_id']}, KSS={info['label']})"
    plot_signal(signal_original, channel_name, title_original, 
                'augmentation_visualizations/01_original_signal.png')

    title_noise = f"Signal + Gaussian Noise (STD={Config.AUG_GAUSSIAN_NOISE_STD}) - {channel_name} Channel"
    plot_signal(signal_with_noise_np, channel_name, title_noise,
                'augmentation_visualizations/02_gaussian_noise.png')

    title_scale = f"Signal + Amplitude Scaling ({scale:.3f}x) - {channel_name} Channel"
    plot_signal(signal_with_scale_np, channel_name, title_scale,
                'augmentation_visualizations/03_amplitude_scaling.png')

    # Print perbandingan statistik
    print("\n" + "="*70)
    print("📊 STATISTIK PERBANDINGAN")
    print("="*70)
    print(f"{'Metric':<20} {'Original':>15} {'+ Noise':>15} {'+ Scale':>15}")
    print("-"*70)
    print(f"{'Mean':<20} {np.mean(signal_original):>15.4f} {np.mean(signal_with_noise_np):>15.4f} {np.mean(signal_with_scale_np):>15.4f}")
    print(f"{'Std Dev':<20} {np.std(signal_original):>15.4f} {np.std(signal_with_noise_np):>15.4f} {np.std(signal_with_scale_np):>15.4f}")
    print(f"{'Min':<20} {np.min(signal_original):>15.4f} {np.min(signal_with_noise_np):>15.4f} {np.min(signal_with_scale_np):>15.4f}")
    print(f"{'Max':<20} {np.max(signal_original):>15.4f} {np.max(signal_with_noise_np):>15.4f} {np.max(signal_with_scale_np):>15.4f}")
    print("="*70 + "\n")

    print("📁 Semua plot tersimpan di folder: augmentation_visualizations/")
    print("   ├─ 01_original_signal.png")
    print("   ├─ 02_gaussian_noise.png")
    print("   └─ 03_amplitude_scaling.png\n")

if __name__ == "__main__":
    visualize_augmentation_samples()
