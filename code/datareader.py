import os
import random
import numpy as np
import torch
import pandas as pd
import mne
from collections import defaultdict, Counter
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from config import Config

# Random Seed untuk Reproducibility
random.seed(22); np.random.seed(22); torch.manual_seed(22)

# Label helpers 
def kss_to_class(kss: int) -> int:
    if Config.NUM_CLASSES == 2:
        return 0 if kss <= 5 else 1
    return 0 if kss <= 3 else (1 if kss <= 6 else 2)

def class_name(class_idx: int) -> str:
    names = ['Alert', 'Drowsy'] if Config.NUM_CLASSES == 2 else ['Alert', 'Low Vigilance', 'Drowsy']
    return names[class_idx]

# Augmentation 

class EEGAugmentation:
    def __init__(self, std=Config.AUG_GAUSSIAN_NOISE_STD,
                 amp_range=Config.AUG_AMPLITUDE_SCALE_RANGE, prob=0.5):
        self.std = std; self.amp_range = amp_range; self.prob = prob

    def __call__(self, signal: torch.Tensor) -> torch.Tensor:
        if np.random.rand() < self.prob:
            signal = signal + torch.randn_like(signal) * self.std
        if np.random.rand() < self.prob:
            signal = signal * np.random.uniform(*self.amp_range)
        return signal

# Dataset 

CHANNELS = ['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H']

class EEGDataset(Dataset):
    def __init__(self, data_dir, csv_path, fold, split, n_splits,
                 window_sec, stride_sec, use_augmentation=False):
        self.data_dir         = data_dir
        self.window_sec       = window_sec
        self.stride_sec       = stride_sec
        self.use_augmentation = use_augmentation
        self.augment          = EEGAugmentation() if use_augmentation else None

        df = pd.read_csv(csv_path)
        if 'subject_id' not in df.columns:
            df['subject_id'] = df['filename'].apply(lambda x: x.split('-')[0])

        splits = list(GroupKFold(n_splits).split(df, df['label'], groups=df['subject_id']))
        idx = splits[fold][0] if split == 'train' else splits[fold][1]
        file_list = df.iloc[idx][['filename', 'label']].values.tolist()

        self.samples = []
        print(f"[{split.upper()}] Indexing (window={window_sec}s, stride={stride_sec}s)...")
        for filename, kss in file_list:
            if kss == 0:
                continue
            file_path = os.path.join(data_dir, filename)
            if not os.path.exists(file_path):
                print(f"  [SKIP] {filename}: not found")
                continue
            try:
                dur = mne.io.read_raw_edf(file_path, preload=False, verbose='error').times[-1]
                if dur <= window_sec:
                    continue
                subject_id = filename.split('-')[0]
                for start in np.arange(0, dur - window_sec, stride_sec):
                    self.samples.append({
                        'file_path': file_path, 'start_sec': float(start),
                        'label': kss, 'class_idx': kss_to_class(kss),
                        'subject_id': subject_id
                    })
            except Exception as e:
                print(f"  [ERROR] {filename}: {e}")

        dist = Counter(class_name(s['class_idx']) for s in self.samples)
        print(f"  Total: {len(self.samples)} windows | {dict(dist)}\n")

        # Subject-level normalization stats
        self.subject_stats = {}
        if getattr(Config, 'NORMALIZATION', 'window') == 'subject':
            self._compute_subject_stats()

    def _compute_subject_stats(self):
        print("[NORM] Computing per-subject stats...")
        files_per_subject = defaultdict(set)
        for s in self.samples:
            files_per_subject[s['subject_id']].add(s['file_path'])

        for subj, fps in files_per_subject.items():
            sum_ch = np.zeros((Config.IN_CHANNELS, 1), dtype=np.float64)
            sumsq  = np.zeros((Config.IN_CHANNELS, 1), dtype=np.float64)
            count  = 0
            for fp in fps:
                try:
                    raw = mne.io.read_raw_edf(fp, preload=True, verbose='error')
                    raw.pick(CHANNELS)
                    sig = raw.get_data() * 1e6
                    sum_ch += sig.sum(axis=1, keepdims=True)
                    sumsq  += (sig ** 2).sum(axis=1, keepdims=True)
                    count  += sig.shape[1]
                except Exception as e:
                    print(f"  [NORM-ERROR] {subj} {fp}: {e}")
            if count > 0:
                mean = sum_ch / count
                std  = np.sqrt(np.maximum(sumsq / count - mean ** 2, 1e-12))
                self.subject_stats[subj] = (mean, std)
            else:
                self.subject_stats[subj] = (np.zeros((Config.IN_CHANNELS, 1)),
                                            np.ones((Config.IN_CHANNELS, 1)))
        print(f"[NORM] Done. {len(self.subject_stats)} subjects.\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        info = self.samples[idx]
        try:
            raw = mne.io.read_raw_edf(info['file_path'], preload=True, verbose='error')
            raw.pick(CHANNELS)
            raw.crop(tmin=info['start_sec'], tmax=info['start_sec'] + self.window_sec, include_tmax=False)
            signal = raw.get_data() * 1e6  # V → μV

            # Normalisasi
            if getattr(Config, 'NORMALIZATION', 'window') == 'subject':
                stats = self.subject_stats.get(info['subject_id'])
                mean, std = stats if stats is not None else (
                    np.mean(signal, axis=1, keepdims=True),
                    np.std(signal, axis=1, keepdims=True)
                )
            else:
                mean = np.mean(signal, axis=1, keepdims=True)
                std  = np.std(signal, axis=1, keepdims=True)
            signal = (signal - mean) / (std + 1e-6)

            signal = torch.tensor(signal, dtype=torch.float32)
            if self.augment:
                signal = self.augment(signal)

            if Config.is_classification():
                return signal, torch.tensor(info['class_idx'], dtype=torch.long)
            else:
                return signal, torch.tensor([info['label'] / Config.KSS_MAX], dtype=torch.float32)

        except Exception:
            sr = Config.SAMPLE_RATE
            dummy_len = int(self.window_sec * sr)
            return torch.zeros((Config.IN_CHANNELS, dummy_len)), torch.tensor([0.0], dtype=torch.float32)


def collate_fn(batch):
    signals, labels = zip(*batch)
    return torch.stack(signals), torch.stack(labels)

# Visualisasi

def visualize_class_comparison(dataset, n_examples=2, save_path='sample_comparison.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_classes  = Config.NUM_CLASSES
    window_sec = dataset.window_sec
    sr         = Config.SAMPLE_RATE

    idx_per_class = defaultdict(list)
    for i, s in enumerate(dataset.samples):
        idx_per_class[s['class_idx']].append(i)

    chosen = {c: random.sample(idx_per_class[c], min(n_examples, len(idx_per_class[c])))
              for c in range(n_classes)}

    n_ch = len(CHANNELS)
    total_cols = n_examples * n_classes
    fig, axes = plt.subplots(n_ch, total_cols, figsize=(total_cols * 5, n_ch * 1.8 + 1.5), sharex=True)
    if total_cols == 1: axes = axes.reshape(-1, 1)
    if n_ch == 1:       axes = axes.reshape(1, -1)

    t = np.linspace(0, window_sec, int(window_sec * sr))
    colors = {0: '#1565C0', 1: '#FF6F00', 2: '#B71C1C'}

    col = 0
    for c in range(n_classes):
        for ex, si in enumerate(chosen[c]):
            info = dataset.samples[si]
            try:
                raw = mne.io.read_raw_edf(info['file_path'], preload=True, verbose='error')
                raw.pick(CHANNELS)
                raw.crop(tmin=info['start_sec'], tmax=info['start_sec'] + window_sec, include_tmax=False)
                sig = raw.get_data() * 1e6
            except Exception as e:
                print(f"[VIZ] Error: {e}"); col += 1; continue

            axes[0, col].set_title(f"{class_name(c)} #{ex+1}\nKSS={info['label']} t={info['start_sec']:.0f}s",
                                   fontsize=9, fontweight='bold', color=colors.get(c, '#424242'))
            for ch in range(n_ch):
                ax = axes[ch, col]
                ax.plot(t, sig[ch], color=colors.get(c, '#424242'), linewidth=0.5, alpha=0.9)
                ax.axhline(0, color='gray', linewidth=0.3, linestyle='--')
                if col == 0: ax.set_ylabel(CHANNELS[ch], fontsize=8, rotation=0, labelpad=30, va='center')
                ymax = min(max(np.abs(sig[ch]).max() * 1.15, 15), 150)
                ax.set_ylim(-ymax, ymax)
                ax.tick_params(labelsize=6)
                if ch == n_ch - 1: ax.set_xlabel('Waktu (s)', fontsize=7)
            col += 1

    fig.suptitle(f'EEG/EOG Signal — {n_classes}-Class | window={window_sec}s | {sr}Hz',
                 fontsize=11, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0.045, 0, 1, 1])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[VIZ] Saved → {save_path}")


# Main (Visualize)

if __name__ == "__main__":
    for split, aug in [('train', False), ('val', False)]:
        ds = EEGDataset('psg', 'label/labels.csv', fold=0, split=split,
                        n_splits=5, window_sec=Config.WINDOW_SEC,
                        stride_sec=Config.STRIDE_SEC if split == 'train' else Config.WINDOW_SEC,
                        use_augmentation=aug)
        print(f"{split.upper()}: {len(ds)} windows")

    visualize_class_comparison(ds, n_examples=1,
                               save_path=f'sample_comparison_{Config.NUM_CLASSES}class.png')