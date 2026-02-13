import numpy as np
import os
import torch
import pandas as pd
import mne
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from config import Config

# Set seed untuk reproduktibilitas
RANDOM_SEED = 2004
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class EEGAugmentation:
    """Modul augmentasi sinyal EEG dasar"""
    def __init__(self, gaussian_std=Config.AUG_GAUSSIAN_NOISE_STD, amplitude_range=Config.AUG_AMPLITUDE_SCALE_RANGE, time_shift_max=Config.AUG_TIME_SHIFT_MAX, prob=0.5):
        self.gaussian_std = gaussian_std
        self.amplitude_range = amplitude_range
        self.time_shift_max = time_shift_max
        self.prob = prob
    
    def add_gaussian_noise(self, signal):
        if np.random.rand() < self.prob:
            noise = torch.randn_like(signal) * self.gaussian_std
            signal = signal + noise
        return signal
    
    def amplitude_scaling(self, signal):
        if np.random.rand() < self.prob:
            scale = np.random.uniform(*self.amplitude_range)
            signal = signal * scale
        return signal
    
    def time_shift(self, signal):
        if np.random.rand() < self.prob:
            shift = np.random.randint(-self.time_shift_max, self.time_shift_max)
            signal = torch.roll(signal, shifts=shift, dims=1)
        return signal

class EEGDataset(Dataset):
    def __init__(self, data_dir, csv_path, fold, split, n_splits, window_sec, stride_sec, use_augmentation=False):
        self.data_dir = data_dir
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.use_augmentation = use_augmentation
        
        df = pd.read_csv(csv_path)
        if 'subject_id' not in df.columns:
            df['subject_id'] = df['filename'].apply(lambda x: x.split('_')[0])

        # GroupKFold mencegah kebocoran data subjek antara train dan val
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(df, df['label'], groups=df['subject_id']))
        train_idx, val_idx = splits[fold]
        
        selected_idx = train_idx if split == 'train' else val_idx
        self.file_list = df.iloc[selected_idx][['filename', 'label']].values.tolist()
        
        self.samples = []
        print(f"[{split.upper()}] Mengindeks file (Stride={stride_sec}s)...")
        
        # Hitung titik potong (sliding window) untuk setiap file rekaman
        for filename, label in self.file_list:
            file_path = os.path.join(self.data_dir, filename)
            if not os.path.exists(file_path):
                continue
                
            try:
                raw_info = mne.io.read_raw_edf(file_path, preload=False, verbose='error')
                duration_sec = raw_info.times[-1]
                max_start = duration_sec - self.window_sec
                
                if max_start > 0:
                    starts = np.arange(0, max_start, self.stride_sec)
                    for start in starts:
                        self.samples.append({
                            'file_path': file_path,
                            'start_sec': start,
                            'label': label
                        })
            except Exception:
                pass

        print(f"Total {split.upper()} Samples Terkumpul: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        file_path = sample_info['file_path']
        start_sec = sample_info['start_sec']
        raw_label = sample_info['label']
        
        # Pemetaan KSS (1-9) ke Kelas Model (0-2)
        if raw_label <= 3:
            label = 0   # Alert
        elif raw_label <= 6:
            label = 1   # Low Vigilance
        else:
            label = 2   # Drowsy
        
        try:
            # Load dan potong sinyal EEG
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose='error')
            target_channels = ['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H']
            raw.pick(target_channels)
            
            t_end = start_sec + self.window_sec
            raw.crop(tmin=start_sec, tmax=t_end, include_tmax=False)
            
            signal = raw.get_data() * 1e6  # Konversi ke mikrovolt
            
            # Z-Score Normalization
            mean = np.mean(signal, axis=1, keepdims=True)
            std = np.std(signal, axis=1, keepdims=True)
            signal = (signal - mean) / (std + 1e-6)
            
            signal = torch.tensor(signal, dtype=torch.float32)
            
            if self.use_augmentation:
                aug = EEGAugmentation()
                signal = aug.add_gaussian_noise(signal)
                signal = aug.amplitude_scaling(signal)
            
            return signal, torch.tensor(label, dtype=torch.long)

        except Exception:
            # Return dummy zeros jika file error agar training tidak crash
            dummy_len = int(self.window_sec * Config.SAMPLE_RATE)
            return torch.zeros((Config.IN_CHANNELS, dummy_len)), torch.tensor(0, dtype=torch.long)

class UniqueRecordingBatchSampler(Sampler):
    """Sampler untuk memastikan setiap batch berisi data dari rekaman yang berbeda"""
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        # Kelompokkan indeks berdasarkan file sumber
        self.file_to_indices = defaultdict(list)
        for idx, sample in enumerate(self.dataset.samples):
            self.file_to_indices[sample['file_path']].append(idx)
            
        self.unique_files = list(self.file_to_indices.keys())
        
        # Penyesuaian jika batch size lebih besar dari jumlah rekaman
        if len(self.unique_files) < self.batch_size:
            print(f"Warning: Rekaman unik ({len(self.unique_files)}) < Batch Size ({self.batch_size}). Menyesuaikan batch size.")
            self.batch_size = len(self.unique_files)

    def __iter__(self):
        # Acak urutan potongan di dalam tiap file
        working_indices = {}
        for file_path in self.unique_files:
            indices = self.file_to_indices[file_path].copy()
            np.random.shuffle(indices)
            working_indices[file_path] = indices
            
        available_files = list(self.unique_files)
        
        # Buat batch dengan mengambil 1 sampel dari tiap file yang terpilih
        while len(available_files) >= self.batch_size:
            selected_files = random.sample(available_files, self.batch_size)
            batch = []
            for f in selected_files:
                batch.append(working_indices[f].pop())
                if len(working_indices[f]) == 0:
                    available_files.remove(f)
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size

def collate_fn(batch):
    signals, labels = zip(*batch)
    signals = torch.stack(signals)
    labels = torch.stack(labels)
    return signals, labels

if __name__ == "__main__":
    print("Testing DataReader...")
    dataset = EEGDataset(
        data_dir='psg',        
        csv_path='label/labels.csv', 
        fold=0,
        split='train',
        n_splits=5,
        window_sec=Config.WINDOW_SEC,
        stride_sec=Config.STRIDE_SEC,
        use_augmentation=False
    )
    
    if len(dataset) > 0:
        sig, lab = dataset[0]
        print(f"Shape Signal: {sig.shape}")
        print(f"Label: {lab}")
    else:
        print("Dataset kosong. Cek path file.")