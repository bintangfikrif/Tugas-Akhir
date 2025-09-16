import numpy as np
import os
from torch.utils.data import Dataset
import torch
import random
import pandas as pd
import mne
import matplotlib.pyplot as plt

# Set random seed for reproducibility
RANDOM_SEED = 2004
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class EEGDataset(Dataset):
    TARGET_CHANNELS = ['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H']

    def __init__(self,
                 data_dir='psg',
                 transform=None,
                 split='train',
                 sample_len=1000  # jumlah sample per channel (window)
                 ):

        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.sample_len = sample_len

        # list file EDF
        self.edf_files = [f for f in os.listdir(data_dir) if f.endswith('.edf')]
        self.edf_files.sort()

        # baca CSV label
        csv_path = os.path.join(os.path.dirname(data_dir), 'label', 'labels.csv')
        df = pd.read_csv(csv_path)
        self.label_dict = dict(zip(df['filename'], df['label']))
        self.labels = [self.label_dict.get(f, None) for f in self.edf_files]

        # pasangkan EDF file dengan label
        all_data = list(zip(self.edf_files, self.labels))

        # split train/val
        total_len = len(all_data)
        train_len = int(0.8 * total_len)
        indices = list(range(total_len))
        random.shuffle(indices)

        train_indices = indices[:train_len]
        val_indices = indices[train_len:]

        if split == 'train':
            self.data = [all_data[i] for i in train_indices]
        elif split == 'val':
            self.data = [all_data[i] for i in val_indices]
        else:
            raise ValueError("Split must be 'train' or 'val'")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        edf_path = os.path.join(self.data_dir, self.data[idx][0])
        label = self.data[idx][1]

        # load EDF
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        picks = mne.pick_channels(raw.ch_names, include=self.TARGET_CHANNELS)

        data, _ = raw[picks, :]
        ch_names = [raw.ch_names[i] for i in picks]

        # pad/truncate tiap channel agar panjangnya sama = sample_len
        signals = []
        for i in range(len(ch_names)):
            sig = data[i]
            if len(sig) >= self.sample_len:
                sig = sig[:self.sample_len]
            else:
                pad_len = self.sample_len - len(sig)
                sig = np.pad(sig, (0, pad_len), mode='constant')
            signals.append(sig)

        signals = np.stack(signals)  # shape: (n_channels, sample_len)
        signals_tensor = torch.tensor(signals, dtype=torch.float32)

        if self.transform:
            signals_tensor = self.transform(signals_tensor)

        return_data = (signals_tensor, label, edf_path, ch_names)
        return return_data


if __name__ == "__main__":
    train_dataset = EEGDataset(split='train')
    val_dataset = EEGDataset(split='val')

    print(f"Train data: {len(train_dataset)}")
    print(f"Val data: {len(val_dataset)}")
    print(f"Total: {len(train_dataset) + len(val_dataset)}")

    # ambil 1 contoh dan plot
    idx = 0
    signals, label, filepath, ch_names = train_dataset[idx]

    print(f"\nContoh data index {idx}")
    print(f"Signals shape: {signals.shape}")  # (7, sample_len)
    print(f"Label (Tingkat kantuk): {label}")
    print(f"File path: {filepath}")

    # Plot setiap channel
    n_channels = signals.shape[0]
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Signals from {os.path.basename(filepath)} | Label (Tingkat Kantuk): {label}", fontsize=14)

    for i in range(n_channels):
        axes[i].plot(signals[i].numpy())
        axes[i].set_ylabel(ch_names[i], rotation=0, labelpad=30)
        axes[i].grid(True)

    axes[-1].set_xlabel("Sample Points")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()
