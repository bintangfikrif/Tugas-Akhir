import numpy as np
import os
import torch
import pandas as pd
import mne
import random
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict
from sklearn.model_selection import GroupKFold
from config import Config

# Set random seeds untuk reproduksibilitas
RANDOM_SEED = 2004
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def parse_pvt_file(csv_path):
    """Membaca file PVT dan mengembalikan list of tuples: (stimulus_time_sec, reaction_time_ms)"""
    events = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()
        
    if not lines:
        return events

    # Format datetime universal untuk file ini: Tahun-Bulan-Tanggal_Jam.Menit.Detik.Milidetik
    time_format = "%Y-%m-%d_%H.%M.%S.%f"

    # Baca waktu mulai (Baris 1)
    base_time_str = lines[0].strip()
    try:
        base_time = datetime.strptime(base_time_str, time_format)
    except ValueError:
        return events # Skip jika format header salah

    # Baca kejadian stimulus & respons (Baris 2 dst)
    for line in lines[1:]:
        line = line.strip()
        if not line: continue
        
        parts = line.split(';')
        if len(parts) == 2:
            stim_str, resp_str = parts
            
            try:
                stim_dt = datetime.strptime(stim_str, time_format)
                resp_dt = datetime.strptime(resp_str, time_format)
                
                # Hitung selisih waktu
                stim_sec = (stim_dt - base_time).total_seconds()
                rt_ms = (resp_dt - stim_dt).total_seconds() * 1000.0
                
                events.append((stim_sec, rt_ms))
            except ValueError:
                continue
                
    return events

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
        # FIX: Gunakan stride_sec yang diberikan, jangan di-override
        self.stride_sec = stride_sec
        
        df = pd.read_csv(csv_path)
        if 'subject_id' not in df.columns:
            df['subject_id'] = df['filename'].apply(lambda x: x.split('_')[0])

        # Gunakan GroupKFold untuk memastikan rekaman dari subjek
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(df, df['label'], groups=df['subject_id']))
        train_idx, val_idx = splits[fold]
        
        selected_idx = train_idx if split == 'train' else val_idx
        self.file_list = df.iloc[selected_idx][['filename', 'label']].values.tolist()
        
        self.samples = []
        
        # Tracking statistik filter
        total_windows = 0
        kept_windows = 0
        
        print(f"[{split.upper()}] Menyaring Data PVT & Mengindeks File (Stride={self.stride_sec}s)...")
        
        # Iterasi setiap file untuk menentukan jendela mana yang akan diambil berdasarkan aturan PVT
        for filename, label in self.file_list:
            file_path = os.path.join(self.data_dir, filename)
            pvt_filename = filename.replace('.edf', '.csv')
            pvt_path = os.path.join('pvt-rt', pvt_filename)
            
            if not os.path.exists(file_path) or not os.path.exists(pvt_path):
                continue
                
            try:
                # Parse data PVT
                pvt_events = parse_pvt_file(pvt_path)
                
                raw_info = mne.io.read_raw_edf(file_path, preload=False, verbose='error')
                duration_sec = raw_info.times[-1]
                max_start = duration_sec - self.window_sec
                
                if max_start > 0:
                    starts = np.arange(0, max_start, self.stride_sec)
                    for start in starts:
                        end = start + self.window_sec
                        total_windows += 1
                        
                        # Ambil kejadian PVT di dalam jendela 30 detik ini
                        window_events = [rt_ms for (stim_sec, rt_ms) in pvt_events if start <= stim_sec < end]
                        
                        keep = False
                        if not window_events:
                            keep = True
                        else:
                            # Hitung rata-rata waktu reaksi (Average RT)
                            avg_rt = sum(window_events) / len(window_events)
                            # Hitung jumlah microsleep (Lapse)
                            lapses = sum(1 for rt in window_events if rt > 500.0)
                            
                            if label <= 3:      
                                keep = (lapses == 0) and (avg_rt < 350.0)
                            
                            elif label >= 7:    
                                keep = (lapses >= 1) or (avg_rt >= 350.0)
                            
                            else:               
                                keep = True
                                
                        if keep:
                            self.samples.append({
                                'file_path': file_path,
                                'start_sec': start,
                                'label': label
                            })
                            kept_windows += 1
                            
            except Exception as e:
                pass

        print(f"Total Jendela Terbentuk: {total_windows} | Jendela Bersih (Diambil): {kept_windows} | Dibuang: {total_windows - kept_windows}")

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