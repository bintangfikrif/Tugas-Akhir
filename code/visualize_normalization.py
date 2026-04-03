import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import pandas as pd
import mne
from config import Config
from datareader import EEGDataset, apply_bandpass

def plot_signal_comparison(signal_before, signal_after, channels, title="Signal Comparison"):
    """
    Plot sinyal sebelum dan sesudah normalisasi untuk beberapa channel.
    """
    n_channels = len(channels)
    fig, axes = plt.subplots(n_channels, 2, figsize=(15, 3*n_channels), sharex=True)

    if n_channels == 1:
        axes = [axes]

    for i, ch in enumerate(channels):
        # Plot sebelum normalisasi
        axes[i][0].plot(signal_before[i], label=f'{ch} (Raw)', alpha=0.7)
        axes[i][0].set_title(f'{ch} - Before Normalization')
        axes[i][0].set_ylabel('Amplitude (μV)')
        axes[i][0].grid(True, alpha=0.3)

        # Plot sesudah normalisasi
        axes[i][1].plot(signal_after[i], label=f'{ch} (Normalized)', color='orange', alpha=0.7)
        axes[i][1].set_title(f'{ch} - After Normalization')
        axes[i][1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('signal_normalization_comparison.png', dpi=300, bbox_inches='tight')
    print("Plot disimpan sebagai 'signal_normalization_comparison.png'")
    plt.show()

def visualize_normalization_sample():
    """
    Visualisasikan satu sample sinyal sebelum dan sesudah normalisasi.
    """
    # Konfigurasi dataset (gunakan parameter yang sama seperti di train.py)
    dataset = EEGDataset(
        data_dir=Config.DATA_DIR,
        csv_path=os.path.join('label/labels.csv'),
        fold=0,
        split='train',
        n_splits=Config.N_SPLITS,
        window_sec=Config.WINDOW_SEC,
        stride_sec=Config.STRIDE_SEC,
        use_augmentation=False  # Matikan augmentasi untuk visualisasi
    )

    if len(dataset) == 0:
        print("Dataset kosong!")
        return

    # Ambil sample pertama
    idx = 0
    info = dataset.samples[idx]
    file_path = info['file_path']
    start_sec = info['start_sec']

    print(f"Visualisasi sample: {os.path.basename(file_path)}, start={start_sec:.1f}s")

    # Baca sinyal mentah (sebelum normalisasi)
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

    # Channel names
    channels = ['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H']

    # Plot comparison
    title = f"EEG Signal Normalization - Subject {info['subject_id']}, KSS={info['label']}, Mode={norm_mode}"
    plot_signal_comparison(signal_raw, signal_normalized, channels[:4], title)  # Tampilkan 4 channel pertama saja

    # Print statistik
    print("\nStatistik sinyal:")
    print("Channel | Raw Mean | Raw Std | Norm Mean | Norm Std")
    print("-" * 55)
    for i, ch in enumerate(channels):
        raw_mean = np.mean(signal_raw[i])
        raw_std = np.std(signal_raw[i])
        norm_mean = np.mean(signal_normalized[i])
        norm_std = np.std(signal_normalized[i])
        print(f"{ch:7} | {raw_mean:8.2f} | {raw_std:7.2f} | {norm_mean:9.4f} | {norm_std:8.4f}")

if __name__ == "__main__":
    visualize_normalization_sample()