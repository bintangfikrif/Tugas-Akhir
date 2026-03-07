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

# Set random seeds untuk reproduksibilitas
RANDOM_SEED = 2004
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# Augmentasi EEG
class EEGAugmentation:
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
        self.use_augmentation = use_augmentation
        # FIX: Gunakan stride_sec yang dikirim, jangan di-override hardcoded
        self.stride_sec = stride_sec
        
        df = pd.read_csv(csv_path)
        if 'subject_id' not in df.columns:
            # DROZY format: "1-1.edf" -> subject_id = "1"
            df['subject_id'] = df['filename'].apply(lambda x: x.split('-')[0])

        # Gunakan GroupKFold untuk memastikan rekaman dari subjek
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(df, df['label'], groups=df['subject_id']))
        train_idx, val_idx = splits[fold]
        
        selected_idx = train_idx if split == 'train' else val_idx
        self.file_list = df.iloc[selected_idx][['filename', 'label']].values.tolist()
        
        self.samples = []
        total_windows = 0

        print(f"[{split.upper()}] Mengindeks File (Window={self.window_sec}s, Stride={self.stride_sec}s)...")

        # Skenario windowing bersih tanpa filter PVT:
        # Alasan menghapus filter PVT:
        # 1. KSS dicatat SEBELUM sesi dimulai — memfilter berdasarkan hasil PVT
        #    menciptakan bias seleksi yang bertentangan dengan label
        # 2. Window tanpa stimulus PVT (mayoritas) selalu lolos filter -> tidak konsisten
        # 3. Dependency pada pvt-rt/*.csv menyebabkan rekaman di-skip diam-diam
        #
        # Skenario pengganti: ambil seluruh rekaman dengan sliding window,
        # buang hanya 30 detik PERTAMA (stabilisasi awal subjek)
        SKIP_FIRST_SEC = 30  # Buang 30 detik pertama tiap rekaman

        for filename, label in self.file_list:
            # Skip rekaman dengan KSS=0 (tidak valid, hanya ada di 7-1.edf)
            if label == 0:
                print(f"  [SKIP] {filename}: KSS=0 tidak valid")
                continue

            file_path = os.path.join(self.data_dir, filename)
            if not os.path.exists(file_path):
                print(f"  [SKIP] {filename}: file tidak ditemukan")
                continue

            try:
                raw_info = mne.io.read_raw_edf(file_path, preload=False, verbose='error')
                duration_sec = raw_info.times[-1]

                # Mulai dari detik ke-30 (lewati stabilisasi awal)
                # Akhiri sehingga window terakhir tidak melebihi durasi
                effective_start = SKIP_FIRST_SEC
                max_start = duration_sec - self.window_sec

                if max_start <= effective_start:
                    print(f"  [SKIP] {filename}: rekaman terlalu pendek ({duration_sec:.1f}s)")
                    continue

                starts = np.arange(effective_start, max_start, self.stride_sec)
                for start in starts:
                    self.samples.append({
                        'file_path': file_path,
                        'start_sec': float(start),
                        'label': label
                    })
                    total_windows += 1

            except Exception as e:
                print(f"  [ERROR] {filename}: {e}")

        # Hitung distribusi kelas hasil windowing
        from collections import Counter
        def kss_to_class(k):
            if k <= 3: return 'Alert'
            elif k <= 6: return 'Low Vigilance'
            else: return 'Drowsy'
        dist = Counter(kss_to_class(s['label']) for s in self.samples)
        print(f"Total Window: {total_windows} | Distribusi: {dict(dist)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        file_path = sample_info['file_path']
        start_sec = sample_info['start_sec']
        raw_label = sample_info['label']
        
        if raw_label <= 3: label = 0   
        elif raw_label <= 6: label = 1   
        else: label = 2   
        
        try:
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose='error')
            target_channels = ['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H']
            raw.pick(target_channels)
            
            t_end = start_sec + self.window_sec
            raw.crop(tmin=start_sec, tmax=t_end, include_tmax=False)
            
            signal = raw.get_data() * 1e6  
            
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
            dummy_len = int(self.window_sec * Config.SAMPLE_RATE)
            return torch.zeros((Config.IN_CHANNELS, dummy_len)), torch.tensor(0, dtype=torch.long)

# Sampler untuk memastikan batch berisi rekaman unik
class UniqueRecordingBatchSampler(Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        
        self.file_to_indices = defaultdict(list)
        for idx, sample in enumerate(self.dataset.samples):
            self.file_to_indices[sample['file_path']].append(idx)
            
        self.unique_files = list(self.file_to_indices.keys())
        
        if len(self.unique_files) < self.batch_size:
            print(f"Warning: Rekaman unik ({len(self.unique_files)}) < Batch Size ({self.batch_size}). Menyesuaikan batch size.")
            self.batch_size = len(self.unique_files)

    def __iter__(self):
        working_indices = {}
        for file_path in self.unique_files:
            indices = self.file_to_indices[file_path].copy()
            np.random.shuffle(indices)
            working_indices[file_path] = indices
            
        available_files = list(self.unique_files)
        
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

# Fungsi collate untuk DataLoader
def collate_fn(batch):
    signals, labels = zip(*batch)
    signals = torch.stack(signals)
    labels = torch.stack(labels)
    return signals, labels

if __name__ == "__main__":
    print("Testing DataReader Hibrida (KSS + Filter PVT)...")
    
    print("\n--- MENGUJI DATA TRAINING (FOLD 0) ---")
    train_dataset = EEGDataset(
        data_dir='psg',        
        csv_path='label/labels.csv', 
        fold=0,
        split='train',
        n_splits=5,
        window_sec=Config.WINDOW_SEC,
        stride_sec=5, # Stride 5 detik untuk perbanyak data training
        use_augmentation=False
    )
    
    print("\n" + "="*40)
    print(f"TOTAL DATA BERSIH (TRAIN) : {len(train_dataset)} jendela")
    print("="*40)
    
    # Menghitung distribusi label secara manual
    label_counts = {0: 0, 1: 0, 2: 0}
    for sample in train_dataset.samples:
        raw_label = sample['label']
        if raw_label <= 3: 
            label_counts[0] += 1
        elif raw_label <= 6: 
            label_counts[1] += 1
        else: 
            label_counts[2] += 1
            
    print("DISTRIBUSI LABEL:")
    print(f"Alert (0)         : {label_counts[0]} sampel")
    print(f"Low Vigilance (1) : {label_counts[1]} sampel")
    print(f"Drowsy (2)        : {label_counts[2]} sampel")
    print("="*40)
    
    # Cek shape tensor jika data tidak kosong
    if len(train_dataset) > 0:
        print("\nMemuat 1 sampel data untuk cek dimensi...")
        sig, lab = train_dataset[0]
        print(f"Shape Signal Train : {sig.shape}")
        print(f"Contoh Label       : {lab}")