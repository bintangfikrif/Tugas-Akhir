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

BANDPASS_LOW  = 0.5   # Hz
BANDPASS_HIGH = 40.0  # Hz

def apply_bandpass(signal_uv: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Terapkan bandpass filter Butterworth orde 4 pada sinyal EEG/EOG.

    Args:
        signal_uv : np.ndarray shape (n_channels, n_times), dalam μV
        sfreq     : sampling frequency (Hz)

    Returns:
        sinyal yang sudah difilter, shape sama dengan input
    """
    from scipy.signal import butter, filtfilt

    nyq  = sfreq / 2.0
    low  = BANDPASS_LOW  / nyq
    high = BANDPASS_HIGH / nyq

    # Butterworth orde 4 — zero-phase (filtfilt) agar tidak ada phase shift
    b, a = butter(4, [low, high], btype='band')

    filtered = np.zeros_like(signal_uv)
    for ch in range(signal_uv.shape[0]):
        filtered[ch] = filtfilt(b, a, signal_uv[ch])

    return filtered

# ============================================================
# FUNGSI KONVERSI LABEL
# Dikontrol oleh Config.NUM_CLASSES:
#   NUM_CLASSES = 3 → Alert(0) / Low Vigilance(1) / Drowsy(2)
#   NUM_CLASSES = 2 → Alert(0) / Drowsy(1)
# ============================================================

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


# ============================================================
# AUGMENTASI EEG
# ============================================================

class EEGAugmentation:
    def __init__(
        self,
        gaussian_std=Config.AUG_GAUSSIAN_NOISE_STD,
        amplitude_range=Config.AUG_AMPLITUDE_SCALE_RANGE,
        time_shift_max=Config.AUG_TIME_SHIFT_MAX,
        prob=0.5
    ):
        self.gaussian_std    = gaussian_std
        self.amplitude_range = amplitude_range
        self.time_shift_max  = time_shift_max
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

    def time_shift(self, signal):
        if np.random.rand() < self.prob:
            shift  = np.random.randint(-self.time_shift_max, self.time_shift_max)
            signal = torch.roll(signal, shifts=shift, dims=1)
        return signal

# ============================================================
# DATASET
# ============================================================

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
            # DROZY format: "1-1.edf" → subject_id = "1"
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
            # Skip KSS=0 (hanya 7-1.edf, tidak valid)
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

                        if getattr(Config, 'USE_BANDPASS_FILTER', True):
                            signal_fp = apply_bandpass(signal_fp, sfreq=Config.SAMPLE_RATE)

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

    # ----------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info      = self.samples[idx]
        file_path = info['file_path']
        start_sec = info['start_sec']
        class_idx = info['class_idx']

        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose='error')
            raw.pick(['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H'])
            raw.crop(tmin=start_sec, tmax=start_sec + self.window_sec, include_tmax=False)

            signal = raw.get_data() * 1e6            # Volt → μV

            if getattr(Config, 'USE_BANDPASS_FILTER', True):
                signal = apply_bandpass(signal, sfreq=Config.SAMPLE_RATE)

            norm_mode = getattr(Config, 'NORMALIZATION', 'window')
            if norm_mode == 'subject' and hasattr(self, 'subject_stats'):
                subject_id = info.get('subject_id', os.path.basename(file_path).split('-')[0])
                subject_stats = self.subject_stats.get(subject_id)
                if subject_stats is not None:
                    mean, std = subject_stats
                    signal = (signal - mean) / (std + 1e-6)
                else:
                    mean = np.mean(signal, axis=1, keepdims=True)
                    std = np.std(signal, axis=1, keepdims=True)
                    signal = (signal - mean) / (std + 1e-6)
            else:
                mean = np.mean(signal, axis=1, keepdims=True)
                std = np.std(signal, axis=1, keepdims=True)
                signal = (signal - mean) / (std + 1e-6)

            signal = torch.tensor(signal, dtype=torch.float32)

            if self.use_augmentation:
                aug    = EEGAugmentation()
                signal = aug.add_gaussian_noise(signal)
                signal = aug.amplitude_scaling(signal)

            return signal, torch.tensor(class_idx, dtype=torch.long)

        except Exception:
            dummy_len = int(self.window_sec * Config.SAMPLE_RATE)
            return (torch.zeros((Config.IN_CHANNELS, dummy_len)),
                    torch.tensor(0, dtype=torch.long))


# ============================================================
# SAMPLER & COLLATE
# ============================================================

class UniqueRecordingBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset    = dataset
        self.batch_size = batch_size

        self.file_to_indices = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            self.file_to_indices[sample['file_path']].append(idx)

        self.unique_files = list(self.file_to_indices.keys())

        if len(self.unique_files) < self.batch_size:
            print(f"Warning: rekaman unik ({len(self.unique_files)}) < "
                  f"batch size ({self.batch_size}). Disesuaikan.")
            self.batch_size = len(self.unique_files)

    def __iter__(self):
        working = {f: self.file_to_indices[f].copy() for f in self.unique_files}
        for lst in working.values():
            np.random.shuffle(lst)

        available = list(self.unique_files)
        while len(available) >= self.batch_size:
            selected = random.sample(available, self.batch_size)
            batch    = []
            for f in selected:
                batch.append(working[f].pop())
                if len(working[f]) == 0:
                    available.remove(f)
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size


def collate_fn(batch):
    signals, labels = zip(*batch)
    return torch.stack(signals), torch.stack(labels)


# ============================================================
# HELPER: muat sinyal mentah μV (tanpa normalisasi, untuk viz)
# ============================================================

CHANNEL_NAMES = ['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H']

def _load_raw_signal(file_path, start_sec, window_sec):
    """Muat sinyal dalam μV tanpa normalisasi, dengan filter opsional."""
    raw = mne.io.read_raw_edf(file_path, preload=True, verbose='error')
    raw.pick(['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H'])
    raw.crop(tmin=start_sec, tmax=start_sec + window_sec, include_tmax=False)
    signal = raw.get_data() * 1e6   # (7, T)

    # Terapkan filter yang sama seperti training agar visualisasi konsisten
    if getattr(Config, 'USE_BANDPASS_FILTER', True):
        signal = apply_bandpass(signal, sfreq=Config.SAMPLE_RATE)

    return signal


# ============================================================
# VISUALISASI 1 — Sinyal mentah per kelas
# ============================================================

def visualize_class_comparison(dataset, n_examples=2, save_path='sample_comparison.png'):
    """
    Tampilkan n_examples window per kelas secara berdampingan.
    Sinyal dalam μV TANPA normalisasi agar mudah dibandingkan secara visual.

    Layout:
        Baris  = channel (7 channel)
        Kolom  = contoh-1 kelas-A | contoh-2 kelas-A | contoh-1 kelas-B | ...
    """
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


# ============================================================
# VISUALISASI 2 — Power Spectral Density per kelas
# ============================================================

def visualize_psd_comparison(dataset, n_samples_per_class=10, save_path='psd_comparison.png'):
    """
    Bandingkan rata-rata PSD antar kelas per channel EEG.

    Theta (4-8 Hz) dan Alpha (8-12 Hz) adalah marker utama kantuk:
    keduanya meningkat saat drowsy.  Plot ini memperlihatkan apakah
    sinyal dalam dataset memang mencerminkan pola tersebut.
    """
    from scipy.signal import welch

    n_classes  = Config.NUM_CLASSES
    window_sec = dataset.window_sec
    sr         = Config.SAMPLE_RATE
    n_ch       = len(CHANNEL_NAMES)

    # Kumpulkan PSD per kelas per channel
    psd_acc = {c: [[] for _ in range(n_ch)] for c in range(n_classes)}

    idx_per_class = defaultdict(list)
    for i, s in enumerate(dataset.samples):
        idx_per_class[s['class_idx']].append(i)

    for c in range(n_classes):
        pool   = idx_per_class[c]
        chosen = random.sample(pool, min(n_samples_per_class, len(pool)))
        for i in chosen:
            info = dataset.samples[i]
            try:
                sig = _load_raw_signal(info['file_path'], info['start_sec'], window_sec)
                for ch in range(n_ch):
                    _, pxx = welch(sig[ch], fs=sr, nperseg=sr * 2)
                    psd_acc[c][ch].append(pxx)
            except Exception:
                continue

    # Referensi frekuensi
    f_ref, _ = welch(np.zeros(int(window_sec * sr)), fs=sr, nperseg=sr * 2)

    # Band shading
    bands = [
        ('δ (1-4)', 1,  4,  '#E3F2FD'),
        ('θ (4-8)', 4,  8,  '#FFF9C4'),
        ('α (8-12)', 8, 12, '#F3E5F5'),
        ('β (12-30)', 12, 30, '#E8F5E9'),
    ]

    colors = {0: '#1565C0', 1: '#FF6F00', 2: '#B71C1C'}

    fig, axes = plt.subplots(nrows=n_ch, ncols=1, figsize=(11, n_ch * 2.5), sharex=True)
    if n_ch == 1:
        axes = [axes]

    for ch in range(n_ch):
        ax = axes[ch]

        # Band shading
        for bname, blo, bhi, bcol in bands:
            ax.axvspan(blo, bhi, alpha=0.18, color=bcol)
            if ch == 0:
                ax.text((blo + bhi) / 2, 1, bname,
                        ha='center', va='bottom', fontsize=6.5,
                        color='#757575', transform=ax.get_xaxis_transform())

        for c in range(n_classes):
            psds = psd_acc[c][ch]
            if len(psds) == 0:
                continue
            arr      = np.array(psds)
            mean_psd = np.mean(arr, axis=0)
            std_psd  = np.std(arr,  axis=0)

            ax.semilogy(f_ref, mean_psd,
                        label=f"{class_name(c)} (n={len(psds)})",
                        color=colors[c], linewidth=1.8)
            ax.fill_between(f_ref,
                            np.maximum(mean_psd - std_psd, 1e-3),
                            mean_psd + std_psd,
                            alpha=0.15, color=colors[c])

        ax.set_xlim(1, 35)
        ax.set_ylabel(f'{CHANNEL_NAMES[ch]}\n(μV²/Hz)', fontsize=7.5)
        ax.tick_params(labelsize=6.5)
        ax.grid(True, alpha=0.25, which='both', linestyle='--')

        if ch == 0:
            ax.legend(fontsize=8, loc='upper right', framealpha=0.8)

    axes[-1].set_xlabel('Frekuensi (Hz)', fontsize=9)
    mode_str = f"{n_classes}-Class"
    fig.suptitle(
        f'Power Spectral Density per Kelas — {mode_str}\n'
        f'Rata-rata {n_samples_per_class} sample/kelas | '
        'Peningkatan θ & α = indikator kantuk',
        fontsize=10, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] PSD tersimpan → {save_path}")


# ============================================================
# VISUALISASI 3 — Per Label (7 channel dalam 1 file per kelas)
# ============================================================

def visualize_labelwise_channel_traces(dataset, n_samples_per_label=5, save_dir='channel_label_plots'):
    """
    Setiap kelas dibuatkan satu file dengan 7 subplot channel.
    Masing-masing channel memplot beberapa sample sinyal dari kelas itu.

    Args:
        dataset: EEGDataset instance
        n_samples_per_label: jumlah sample per kelas untuk setiap channel
        save_dir: direktori untuk menyimpan plot
    """
    os.makedirs(save_dir, exist_ok=True)

    n_classes = Config.NUM_CLASSES
    window_sec = dataset.window_sec
    sr = Config.SAMPLE_RATE
    n_ch = len(CHANNEL_NAMES)

    idx_per_class = defaultdict(list)
    for i, s in enumerate(dataset.samples):
        idx_per_class[s['class_idx']].append(i)

    t = np.linspace(0, window_sec, int(window_sec * sr))
    colors = {0: '#1565C0', 1: '#FF6F00', 2: '#B71C1C'}

    for label_idx in range(n_classes):
        label_name = class_name(label_idx)
        pool = idx_per_class[label_idx]

        if len(pool) == 0:
            print(f"  [SKIP] Label {label_name}: tidak ada sample")
            continue

        chosen = random.sample(pool, min(n_samples_per_label, len(pool)))

        fig, axes = plt.subplots(n_ch, 1, figsize=(12, n_ch * 1.8), sharex=True)
        if n_ch == 1:
            axes = [axes]

        for ch_idx, ch_name in enumerate(CHANNEL_NAMES):
            ax = axes[ch_idx]

            for sample_i in chosen:
                info = dataset.samples[sample_i]
                try:
                    sig = _load_raw_signal(info['file_path'], info['start_sec'], window_sec)
                    ax.plot(t, sig[ch_idx], color=colors.get(label_idx, '#424242'), alpha=0.6, linewidth=1)
                except Exception as e:
                    print(f"    [ERROR] Sample {sample_i} channel {ch_name}: {e}")

            ax.set_ylabel(ch_name, fontsize=9, rotation=0, labelpad=40, va='center')
            ax.grid(True, alpha=0.25)
            ax.axhline(0, color='black', linewidth=0.6, linestyle='--')

            all_y = np.concatenate([line.get_ydata() for line in ax.lines]) if ax.lines else np.array([0])
            y_min, y_max = np.percentile(all_y, [5, 95])
            margin = max((y_max - y_min) * 0.12, 1.0)
            ax.set_ylim(y_min - margin, y_max + margin)
            ax.set_yticks([])

            if ch_idx == 0:
                ax.set_title(f"Channel {ch_name} — Label {label_name} ({len(chosen)} sample)", fontsize=12, fontweight='bold')

        axes[-1].set_xlabel('Time (s)', fontsize=10)

        fig.suptitle(f"Per Label: {label_name} | {n_ch} channel | {len(chosen)} sample per channel", fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0.05, 0.03, 0.98, 0.95])

        save_path = os.path.join(save_dir, f'label_{label_name.replace(" ", "_")}_channel_traces.png')
        plt.savefig(save_path, dpi=180, bbox_inches='tight')
        plt.close(fig)

        print(f"  [VIZ] Simpan: {save_path}")

    print(f"  [VIZ] Selesai: {n_classes} file labelwise di {save_dir}/")


# ============================================================
# MAIN
# ============================================================

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

    # Visualisasi: 1 contoh sinyal EEG+EOG per kelas
    # Jumlah kolom otomatis menyesuaikan NUM_CLASSES (2 atau 3)
    n_classes = Config.NUM_CLASSES
    print(f"\n[VIZ] Membuat plot sinyal — {n_classes} kelas ({mode_str})...")
    visualize_class_comparison(
        train_dataset,
        n_examples=1,           # 1 contoh per kelas
        save_path=f'sample_comparison_{n_classes}class.png'
    )

    print("[VIZ] Membuat plot PSD...")
    visualize_psd_comparison(
        train_dataset,
        n_samples_per_class=15,
        save_path=f'psd_comparison_{n_classes}class.png'
    )

    print(f"\nSelesai. Output:")
    print(f"  sample_comparison_{n_classes}class.png  — sinyal μV per kelas")
    print(f"  psd_comparison_{n_classes}class.png     — PSD theta/alpha per channel")

    # Generate plot per channel per label
    print(f"\n[VIZ] Membuat plot per label (7 channel dalam 1 file per label)...")
    visualize_labelwise_channel_traces(train_dataset, n_samples_per_label=1, save_dir='channel_label_plots')
    print(f"  channel_label_plots/ — {n_classes} file PDF/PNG labels")