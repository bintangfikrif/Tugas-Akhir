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
        # csv_path logic: try relative to provided data_dir, otherwise try repo-level 'label/labels.csv'
        csv_path = os.path.join(os.path.dirname(data_dir), 'label', 'labels.csv')
        if not os.path.exists(csv_path):
            # fallback: label folder expected as sibling of this script's parent
            repo_root = os.path.dirname(os.path.dirname(__file__))
            alt = os.path.join(repo_root, 'label', 'labels.csv')
            if os.path.exists(alt):
                csv_path = alt

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"labels.csv not found at {csv_path}")

        df = pd.read_csv(csv_path)
        self.label_map = {0: 0, 1: 1}
        mapped = []
        for _, row in df.iterrows():
            fname = row['filename']
            lab = row['label'] 
            
            try:
                kss_level = int(lab)
                binary_label = None
                
                # Tentukan Threshold
                # KSS 1-5 = 0 (Tidak Mengantuk)
                # KSS 6-9 = 1 (Mengantuk)
                if 1 <= kss_level <= 5:
                    binary_label = 0
                elif 6 <= kss_level <= 9:
                    binary_label = 1
                
                if binary_label is not None:
                    mapped.append((fname, binary_label))
                    
            except Exception:
                mapped.append((fname, None))

        self.label_dict = dict(mapped)
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

        self.sample_rate = 512  
        self.sample_len = self.window_sec * self.sample_rate 
        self.windows = []

        for fname, label in self.data:
            if label is None:
                continue
                
            edf_path = os.path.join(self.data_dir, fname)
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                if not all(ch in raw.ch_names for ch in self.TARGET_CHANNELS):
                    print(f"Peringatan: Melewatkan {fname} karena kekurangan channel.")
                    continue
                    
                n_samples = raw.n_times
            except Exception as e:
                print(f"Error membaca {fname}: {e}. Melewatkan file.")
                continue

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
        picks = mne.pick_channels(raw.ch_names, include=self.TARGET_CHANNELS, ordered=True)
        data, _ = raw[picks, start:start + self.sample_len]
        
        ch_names = [raw.ch_names[i] for i in picks]
        reordered_data = []
        for target_ch in self.TARGET_CHANNELS:
            try:
                idx_in_raw = ch_names.index(target_ch)
                reordered_data.append(data[idx_in_raw])
            except ValueError:
                raise RuntimeError(f"Channel {target_ch} tidak ditemukan di {edf_path} saat __getitem__")

        signals = np.stack(reordered_data)
        signals_tensor = torch.tensor(signals, dtype=torch.float32)

        if self.transform:
            signals_tensor = self.transform(signals_tensor)

        return signals_tensor, label, edf_path, self.TARGET_CHANNELS, start


if __name__ == "__main__":
    data_dir_path = 'psg' 
    
    if not os.path.exists(data_dir_path):
        print(f"Direktori data '{data_dir_path}' tidak ditemukan.")
        print("Pastikan Anda menjalankan skrip ini dari direktori yang benar,")
        print("atau ubah variabel 'data_dir_path' di dalam `if __name__ == '__main__':`")
    else:
        fold = 0  
        train_dataset = EEGDataset(data_dir=data_dir_path, split='train', fold=fold)
        val_dataset = EEGDataset(data_dir=data_dir_path, split='val', fold=fold)

        print(f"Fold {fold+1}")
        print(f"Train windows: {len(train_dataset)}")
        print(f"Val windows: {len(val_dataset)}")
        print(f"Label map: {train_dataset.label_map}") # Harusnya {0: 0, 1: 1}

        # ambil 1 contoh dan plot
        if len(train_dataset) > 0:
            idx = random.randint(0, len(train_dataset) - 1)
            signals, label, filepath, ch_names, start = train_dataset[idx]

            print(f"\nContoh window index {idx}")
            print(f"Signals shape: {signals.shape}")  # (7, 2560)
            print(f"Label: {label}") # Harusnya 0 atau 1
            print(f"File path: {filepath}")
            print(f"Start index: {start}")
            print(f"Channel names: {ch_names}")

            # Plot sinyal
            n_channels = signals.shape[0]
            fig, axes = plt.subplots(n_channels, 1, figsize=(12, 10), sharex=True)
            title = f"Label: {'Mengantuk' if label == 1 else 'Tidak Mengantuk'} (KSS {'>=6' if label == 1 else '<=5'})"
            fig.suptitle(f"Signals from {os.path.basename(filepath)} | {title} | Start: {start}", fontsize=14)

            for i in range(n_channels):
                axes[i].plot(signals[i].numpy())
                axes[i].set_ylabel(ch_names[i], rotation=0, labelpad=30)
                axes[i].grid(True)

            axes[-1].set_xlabel("Sample Points")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
        else:
            print("\nTidak ada data training untuk di-plot. Cek folder data dan file CSV.")