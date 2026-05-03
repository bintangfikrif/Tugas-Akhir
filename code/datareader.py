import numpy as np
import os
import torch
import pandas as pd
import mne
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict, Counter
from sklearn.model_selection import GroupKFold
from config import Config

# Set random seeds untuk reproduksibilitas
RANDOM_SEED = 22
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def kss_to_class(kss: int) -> int:
    """Konversi nilai KSS mentah ke indeks kelas."""
    if Config.NUM_CLASSES == 2:
        # Binary: Alert = KSS 1-5 (0), Drowsy = KSS 6-9 (1)
        return 0 if kss <= 5 else 1
    else:
        # 3-class: Alert(0), Low Vigilance(1), Drowsy(2)
        if kss <= 3:   return 0
        elif kss <= 6: return 1
        else:          return 2

def class_name(class_idx: int) -> str:
    """Nama kelas berdasarkan indeks."""
    if Config.NUM_CLASSES == 2:
        return ['Alert', 'Drowsy'][class_idx]
    else:
        return ['Alert', 'Low Vigilance', 'Drowsy'][class_idx]

def kss_to_float(kss: int) -> float:
    """Mengembalikan nilai KSS asli sebagai float untuk regresi."""
    return float(kss)

def get_binary_label(kss: int) -> int:
    # Threshold: 1-5 Alert (0), 6-9 Drowsy (1)
    return 0 if kss <= 5 else 1

# AUGMENTASI 
class EEGAugmentation:
    def __init__(
        self,
        gaussian_std=Config.AUG_GAUSSIAN_NOISE_STD,
        amplitude_range=Config.AUG_AMPLITUDE_SCALE_RANGE,
        prob=0.5
    ):
        self.gaussian_std    = gaussian_std
        self.amplitude_range = amplitude_range
        self.prob            = prob

    def add_gaussian_noise(self, signal):
        if np.random.rand() < self.prob:
            noise  = torch.randn_like(signal) * self.gaussian_std
            signal = signal + noise
        return signal

    def amplitude_scaling(self, signal):
        if np.random.rand() < self.prob:
            scale  = np.random.uniform(*self.amplitude_range)
            signal = signal * scale
        return signal

# DATASET
class EEGDataset(Dataset):
    def __init__(
        self,
        data_dir, csv_path,
        fold, split, n_splits,
        window_sec, stride_sec,
        use_augmentation=False
    ):
        self.data_dir         = data_dir
        self.window_sec       = window_sec
        self.stride_sec       = stride_sec
        self.use_augmentation = use_augmentation

        df = pd.read_csv(csv_path)
        if 'subject_id' not in df.columns:
            df['subject_id'] = df['filename'].apply(lambda x: x.split('-')[0])

        gkf    = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(df, df['label'], groups=df['subject_id']))
        train_idx, val_idx = splits[fold]

        selected_idx   = train_idx if split == 'train' else val_idx
        self.file_list = df.iloc[selected_idx][['filename', 'label']].values.tolist()

        self.samples  = []
        total_windows = 0

        mode_str = f"{Config.NUM_CLASSES}-class"
        print(f"[{split.upper()} | {mode_str}] Mengindeks file "
              f"(Window={window_sec}s, Stride={stride_sec}s)...")

        for filename, label in self.file_list:
            if label == 0:
                print(f"  [SKIP] {filename}: KSS=0 tidak valid")
                continue

            file_path = os.path.join(self.data_dir, filename)
            if not os.path.exists(file_path):
                print(f"  [SKIP] {filename}: file tidak ditemukan")
                continue

            try:
                raw_info     = mne.io.read_raw_edf(file_path, preload=False, verbose='error')
                duration_sec = raw_info.times[-1]

                max_start = duration_sec - self.window_sec

                if max_start <= 0:
                    print(f"  [SKIP] {filename}: terlalu pendek ({duration_sec:.1f}s)")
                    continue

                subject_id = filename.split('-')[0]
                for start in np.arange(0, max_start, self.stride_sec):
                    self.samples.append({
                        'file_path':  file_path,
                        'start_sec':  float(start),
                        'label':      label,             # raw KSS (disimpan untuk visualisasi)
                        'class_idx':  kss_to_class(label),
                        'subject_id': subject_id
                    })
                    total_windows += 1

            except Exception as e:
                print(f"  [ERROR] {filename}: {e}")

        dist = Counter(class_name(s['class_idx']) for s in self.samples)
        print(f"Total Window: {total_windows} | Distribusi: {dict(dist)}\n")

        self.subject_stats = {}
        if getattr(Config, 'NORMALIZATION', 'window') == 'subject':
            print("[NORM] Menghitung statistik per-subjek...")
            files_per_subject = defaultdict(set)
            for s in self.samples:
                files_per_subject[s['subject_id']].add(s['file_path'])

            for subject_id, file_paths in files_per_subject.items():
                sum_ch = np.zeros((Config.IN_CHANNELS, 1), dtype=np.float64)
                sumsq_ch = np.zeros((Config.IN_CHANNELS, 1), dtype=np.float64)
                count = 0

                for fp in file_paths:
                    try:
                        raw = mne.io.read_raw_edf(fp, preload=True, verbose='error')
                        raw.pick(['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H'])
                        signal_fp = raw.get_data() * 1e6

                        sum_ch += signal_fp.sum(axis=1, keepdims=True)
                        sumsq_ch += (signal_fp ** 2).sum(axis=1, keepdims=True)
                        count += signal_fp.shape[1]

                    except Exception as e:
                        print(f"  [NORM-ERROR] subject {subject_id} file {fp}: {e}")

                if count > 0:
                    mean = sum_ch / count
                    var = np.maximum(sumsq_ch / count - mean ** 2, 1e-12)
                    std = np.sqrt(var)
                    self.subject_stats[subject_id] = (mean, std)
                else:
                    self.subject_stats[subject_id] = (np.zeros((Config.IN_CHANNELS, 1)), np.ones((Config.IN_CHANNELS, 1)))

            print(f"[NORM] Selesai. subject_count={len(self.subject_stats)}\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info      = self.samples[idx]
        file_path = info['file_path']
        start_sec = info['start_sec']

        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose='error')
            raw.pick(['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H'])
            raw.crop(tmin=start_sec, tmax=start_sec + self.window_sec, include_tmax=False)

            # Downsampling (tetap sama sesuai Config lo)
            if getattr(Config, 'USE_DOWNSAMPLE', False):
                target_sr = Config.DOWNSAMPLE_RATE
                if target_sr != Config.ORIGINAL_SAMPLE_RATE:
                    raw.resample(target_sr, npad='auto')
            
            current_sr = Config.DOWNSAMPLE_RATE if getattr(Config, 'USE_DOWNSAMPLE', False) else Config.ORIGINAL_SAMPLE_RATE
            signal = raw.get_data() * 1e6 # Volt -> μV

            # Normalisasi Sinyal (Subject/Window) tetap sama
            norm_mode = getattr(Config, 'NORMALIZATION', 'window')
            if norm_mode == 'subject' and hasattr(self, 'subject_stats'):
                subject_id = info.get('subject_id', os.path.basename(file_path).split('-')[0])
                subject_stats = self.subject_stats.get(subject_id)
                if subject_stats is not None:
                    mean, std = subject_stats
                    signal = (signal - mean) / (std + 1e-6)
                else:
                    signal = (signal - np.mean(signal, axis=1, keepdims=True)) / (np.std(signal, axis=1, keepdims=True) + 1e-6)
            else:
                signal = (signal - np.mean(signal, axis=1, keepdims=True)) / (np.std(signal, axis=1, keepdims=True) + 1e-6)

            signal = torch.tensor(signal, dtype=torch.float32)

            if self.use_augmentation:
                aug = EEGAugmentation()
                signal = aug.add_gaussian_noise(signal)
                signal = aug.amplitude_scaling(signal)

            task_type = getattr(Config, 'TASK_TYPE', 'regression').lower()

            if task_type == 'classification':
                label = info['class_idx']
                return signal, torch.tensor(label, dtype=torch.long)
            else:
                # REGRESSION: Label tetap skor KSS dinormalisasi ke 0-1
                kss_raw = float(info['label']) # Ambil skor KSS 1-9
                normalized_kss = kss_raw / Config.KSS_MAX
                return signal, torch.tensor([normalized_kss], dtype=torch.float32)

        except Exception as e:
            # Handle error dengan dummy tensor float
            dummy_len = int(self.window_sec * (Config.DOWNSAMPLE_RATE if Config.USE_DOWNSAMPLE else Config.ORIGINAL_SAMPLE_RATE))
            return (torch.zeros((Config.IN_CHANNELS, dummy_len)), 
                    torch.tensor([0.0], dtype=torch.float32))

def collate_fn(batch):
    signals, labels = zip(*batch)
    return torch.stack(signals), torch.stack(labels)

CHANNEL_NAMES = ['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H']

def _load_raw_signal(file_path, start_sec, window_sec):
    """Muat sinyal dalam μV tanpa normalisasi, dengan filter opsional."""
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose='error')
    raw.pick(['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H'])
    raw.crop(tmin=start_sec, tmax=start_sec + window_sec, include_tmax=False)
    signal = raw.get_data() * 1e6   # (7, T)

    return signal

# VISUALISASI Sinyal mentah per kelas
def visualize_class_comparison(dataset, n_examples=2, save_path='sample_comparison.png'):
    n_classes  = Config.NUM_CLASSES
    window_sec = dataset.window_sec
    sr         = Config.SAMPLE_RATE
    n_ch       = len(CHANNEL_NAMES)

    # Kumpulkan indeks per kelas
    idx_per_class = defaultdict(list)
    for i, s in enumerate(dataset.samples):
        idx_per_class[s['class_idx']].append(i)

    chosen = {}
    for c in range(n_classes):
        pool = idx_per_class[c]
        if len(pool) == 0:
            print(f"[VIZ] Tidak ada sample kelas {class_name(c)}, skip.")
            chosen[c] = []
        else:
            chosen[c] = random.sample(pool, min(n_examples, len(pool)))

    total_cols = n_examples * n_classes
    fig_w      = max(14, total_cols * 5)
    fig_h      = n_ch * 1.8 + 1.5

    fig, axes = plt.subplots(
        nrows=n_ch, ncols=total_cols,
        figsize=(fig_w, fig_h),
        sharex=True
    )
    # Pastikan selalu 2D
    if total_cols == 1:
        axes = axes.reshape(-1, 1)
    if n_ch == 1:
        axes = axes.reshape(1, -1)

    t      = np.linspace(0, window_sec, int(window_sec * sr))
    colors = {0: '#1565C0', 1: '#FF6F00', 2: '#B71C1C'}  # biru tua, amber, merah tua

    col_idx = 0
    for c in range(n_classes):
        color = colors.get(c, '#424242')
        for ex_num, sample_i in enumerate(chosen[c]):
            info = dataset.samples[sample_i]
            try:
                sig = _load_raw_signal(info['file_path'], info['start_sec'], window_sec)
            except Exception as e:
                print(f"[VIZ] Gagal muat sample: {e}")
                col_idx += 1
                continue

            axes[0, col_idx].set_title(
                f"{class_name(c)} — Contoh {ex_num + 1}\n"
                f"KSS={info['label']} | t={info['start_sec']:.0f}s",
                fontsize=9, fontweight='bold', color=color, pad=4
            )

            for ch in range(n_ch):
                ax = axes[ch, col_idx]
                ax.plot(t, sig[ch], color=color, linewidth=0.5, alpha=0.9)
                ax.axhline(0, color='gray', linewidth=0.3, linestyle='--')

                if col_idx == 0:
                    ax.set_ylabel(CHANNEL_NAMES[ch], fontsize=8,
                                  rotation=0, labelpad=30, va='center')

                ax.tick_params(labelsize=6)
                ax.set_xlim(0, window_sec)

                # Skala Y adaptif, dibatasi ±150 μV
                peak = np.abs(sig[ch]).max()
                ymax = min(max(peak * 1.15, 15), 150)
                ax.set_ylim(-ymax, ymax)

                if ch == n_ch - 1:
                    ax.set_xlabel('Waktu (s)', fontsize=7)

            # Garis pemisah antar kelas
            if ex_num == n_examples - 1 and c < n_classes - 1:
                for ch in range(n_ch):
                    axes[ch, col_idx].spines['right'].set_edgecolor('#BDBDBD')
                    axes[ch, col_idx].spines['right'].set_linewidth(2)

            col_idx += 1

    mode_str = f"{n_classes}-Class"
    fig.suptitle(
        f'Perbandingan Sinyal EEG/EOG — {mode_str}\n'
        f'Window={window_sec}s | {sr}Hz | 7 Channel | Amplitudo dalam μV (tanpa normalisasi)',
        fontsize=11, fontweight='bold', y=1.01
    )
    plt.tight_layout(rect=[0.045, 0, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Sinyal tersimpan → {save_path}")

# MAIN
if __name__ == "__main__":
    mode_str = f"{Config.NUM_CLASSES}-Class"
    print("=" * 60)
    print(f"  EEGDataset — Mode: {mode_str}")
    print("=" * 60)

    train_dataset = EEGDataset(
        data_dir='psg',
        csv_path='label/labels.csv',
        fold=0, split='train', n_splits=5,
        window_sec=Config.WINDOW_SEC,
        stride_sec=Config.STRIDE_SEC,
        use_augmentation=False
    )

    val_dataset = EEGDataset(
        data_dir='psg',
        csv_path='label/labels.csv',
        fold=0, split='val', n_splits=5,
        window_sec=Config.WINDOW_SEC,
        stride_sec=Config.WINDOW_SEC,
        use_augmentation=False
    )

    print("=" * 60)
    print(f"  TRAIN : {len(train_dataset):>5} window")
    print(f"  VAL   : {len(val_dataset):>5} window")

    print("\n  Distribusi TRAIN:")
    cnt = Counter(class_name(s['class_idx']) for s in train_dataset.samples)
    for cls, n in sorted(cnt.items()):
        print(f"    {cls:<16}: {n}")

    print("\n  Distribusi VAL:")
    cnt_v = Counter(class_name(s['class_idx']) for s in val_dataset.samples)
    for cls, n in sorted(cnt_v.items()):
        print(f"    {cls:<16}: {n}")
    print("=" * 60)

    if len(train_dataset) > 0:
        sig, lab = train_dataset[0]
        print(f"\n  Shape signal : {sig.shape}  (channels × timesteps)")
        print(f"  Label contoh : {lab.item()} = {class_name(lab.item())}")

    # Visualisasi contoh sinyal EEG+EOG per kelas
    n_classes = Config.NUM_CLASSES
    print(f"\n[VIZ] Membuat plot sinyal — {n_classes} kelas ({mode_str})...")
    visualize_class_comparison(
        train_dataset,
        n_examples=1,           # 1 contoh per kelas
        save_path=f'sample_comparison_{n_classes}class.png'
    )