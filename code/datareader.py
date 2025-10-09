import numpy as np
import os
from torch.utils.data import Dataset
import torch
import random
import pandas as pd
import mne
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

# reproducibility
RANDOM_SEED = 2004
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class EEGDataset(Dataset):
    TARGET_CHANNELS = ['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H']

    def __init__(self,
                 data_dir='psg',
                 fold=0,
                 split='train',
                 n_splits=5,
                 window_sec=5, 
                 transform=None,
                 random_offset=True):

        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.window_sec = window_sec
        self.random_offset = random_offset

        # list file EDF
        self.edf_files = [f for f in os.listdir(data_dir) if f.endswith('.edf')]
        self.edf_files.sort()

        # baca CSV label
        csv_path = os.path.join(os.path.dirname(data_dir), 'label', 'labels.csv')
        df = pd.read_csv(csv_path)
        self.label_dict = dict(zip(df['filename'], df['label']))
        self.labels = [self.label_dict.get(f, None) for f in self.edf_files]

        all_data = list(zip(self.edf_files, self.labels))

        # 5-fold split
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        folds = list(kf.split(all_data))

        train_idx, val_idx = folds[fold]
        if split == 'train':
            self.data = [all_data[i] for i in train_idx]
        elif split == 'val':
            self.data = [all_data[i] for i in val_idx]
        else:
            raise ValueError("Split must be 'train' or 'val'")

        # precompute info setiap EDF (jumlah window)
        self.sample_rate = 512  
        self.sample_len = self.window_sec * self.sample_rate 
        self.windows = []

        for fname, label in self.data:
            edf_path = os.path.join(self.data_dir, fname)
            raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
            picks = mne.pick_channels(raw.ch_names, include=self.TARGET_CHANNELS)
            n_samples = raw.n_times

            # random offset
            offset = random.randint(0, self.sample_rate) if self.random_offset else 0

            # buat daftar window start index
            for start in range(offset, n_samples - self.sample_len, self.sample_len):
                self.windows.append((edf_path, label, start))

        random.shuffle(self.windows)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        edf_path, label, start = self.windows[idx]

        # load EDF
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        picks = mne.pick_channels(raw.ch_names, include=self.TARGET_CHANNELS)
        data, _ = raw[picks, start:start + self.sample_len]
        ch_names = [raw.ch_names[i] for i in picks]

        # konversi ke tensor
        signals = np.stack(data)
        signals_tensor = torch.tensor(signals, dtype=torch.float32)

        if self.transform:
            signals_tensor = self.transform(signals_tensor)

        return signals_tensor, label, edf_path, ch_names, start


if __name__ == "__main__":
    fold = 0  # fold ke-1
    train_dataset = EEGDataset(split='train', fold=fold)
    val_dataset = EEGDataset(split='val', fold=fold)

    print(f"Fold {fold+1}")
    print(f"Train windows: {len(train_dataset)}")
    print(f"Val windows: {len(val_dataset)}")

    # ambil 1 contoh dan plot
    idx = random.randint(0, len(train_dataset) - 1)
    signals, label, filepath, ch_names, start = train_dataset[idx]

    print(f"\nContoh window index {idx}")
    print(f"Signals shape: {signals.shape}")  # (7, 2560)
    print(f"Label: {label}")
    print(f"File path: {filepath}")
    print(f"Start index: {start}")

    # Plot sinyal
    n_channels = signals.shape[0]
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Signals from {os.path.basename(filepath)} | Label: {label} | Start: {start}", fontsize=14)

    for i in range(n_channels):
        axes[i].plot(signals[i].numpy())
        axes[i].set_ylabel(ch_names[i], rotation=0, labelpad=30)
        axes[i].grid(True)

    axes[-1].set_xlabel("Sample Points")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
