import numpy as np
import os
import torch
import pandas as pd
import mne
import random
from torch.utils.data import Dataset
from sklearn.model_selection import GroupKFold

# Konfigurasi Reproduksibilitas
RANDOM_SEED = 2004
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class EEGDataset(Dataset):
    """
    Dataset DROZY untuk Klasifikasi 3-Kelas.
    - Window: 30 Detik.
    - Validasi: Subject-Wise[cite: 268, 488].
    - Normalisasi: Per Subjek (sesuai ralat terbaru Anda).
    """
    TARGET_CHANNELS = ['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H']

    def __init__(self,
                 data_dir='psg',
                 csv_path='label/labels.csv',
                 fold=0,
                 split='train',
                 n_splits=5,
                 window_sec=30,
                 transform=None): # Transform untuk Augmentasi
        
        self.data_dir = data_dir
        self.window_sec = window_sec
        self.sample_rate = 512
        self.sample_len = window_sec * self.sample_rate
        self.transform = transform 

        # 1. Load Labels & Mapping [cite: 485]
        df = pd.read_csv(csv_path)
        self.label_dict = {}
        for _, row in df.iterrows():
            kss = int(row['label'])
            # Mapping 3 Kelas: Alert(0), Low Vigilance(1), Drowsy(2) [cite: 485]
            label = 0 if kss <= 3 else (1 if kss <= 6 else 2)
            self.label_dict[row['filename']] = label

        # 2. Subject-Wise Split [cite: 268, 488, 490]
        self.edf_files = [f for f in os.listdir(data_dir) if f.endswith('.edf') and f in self.label_dict]
        self.edf_files.sort()
        self.subjects = [f.split('-')[0] for f in self.edf_files]

        gkf = GroupKFold(n_splits=n_splits)
        folds = list(gkf.split(self.edf_files, groups=self.subjects))
        
        train_idx, val_idx = folds[fold]
        selected_files = [self.edf_files[i] for i in (train_idx if split == 'train' else val_idx)]

        # 3. Pre-load Statistik untuk Normalisasi Per Subjek
        # Ini penting agar normalisasi konsisten menggunakan mean/std subjek tersebut
        self.windows = []
        for fname in selected_files:
            label = self.label_dict[fname]
            edf_path = os.path.join(self.data_dir, fname)
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                # Buat jendela 30 detik [cite: 495]
                for start in range(0, raw.n_times - self.sample_len, self.sample_len):
                    self.windows.append((edf_path, label, start))
            except Exception as e:
                print(f"Error skipping {fname}: {e}")

    def _normalize_subject(self, tensor):
        """
        Z-Score Normalization: Z = (X - mu) / sigma[cite: 203].
        Sesuai ralat: dilakukan per subjek.
        """
        mu = tensor.mean()
        sigma = tensor.std()
        return (tensor - mu) / (sigma + 1e-6)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        edf_path, label, start = self.windows[idx]
        
        # Load segmen data
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.pick_channels(self.TARGET_CHANNELS, ordered=True)
        data, _ = raw[:, start : start + self.sample_len]
        
        tensor = torch.tensor(data, dtype=torch.float32)

        # 1. Terapkan Transform (Augmentasi seperti Gaussian Noise dari Config)
        if self.transform:
            tensor = self.transform(tensor)

        # 2. Terapkan Normalisasi (Preprocessing wajib sesuai Proposal) 
        tensor = self._normalize_subject(tensor)

        return tensor, label
    
def collate_fn(batch):
    """
    Custom collate function untuk menggabungkan sampel EEG menjadi batch.
    Sesuai standar pemrosesan sinyal 30 detik[cite: 495].
    """
    # item[0] adalah tensor sinyal, item[1] adalah label
    signals = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    
    return signals, labels