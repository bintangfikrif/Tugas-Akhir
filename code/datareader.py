import numpy as np
import os
import torch
import pandas as pd
import mne
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from config import Config

# ==========================================
# 1. KONFIGURASI SEED
# ==========================================
RANDOM_SEED = 2004
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# ==========================================
# 2. KELAS AUGMENTASI DATA
# ==========================================
class EEGAugmentation:
    def __init__(self, 
                 gaussian_std=Config.AUG_GAUSSIAN_NOISE_STD,
                 amplitude_range=Config.AUG_AMPLITUDE_SCALE_RANGE,
                 time_shift_max=Config.AUG_TIME_SHIFT_MAX,
                 prob=0.5):
        self.gaussian_std = gaussian_std
        self.amplitude_range = amplitude_range
        self.time_shift_max = time_shift_max
        self.prob = prob
    
    def add_gaussian_noise(self, signal):
        """Menambahkan noise putih acak ke sinyal"""
        if np.random.rand() < self.prob:
            noise = torch.randn_like(signal) * self.gaussian_std
            signal = signal + noise
        return signal
    
    def amplitude_scaling(self, signal):
        """Mengalikan sinyal dengan faktor acak"""
        if np.random.rand() < self.prob:
            scale = np.random.uniform(*self.amplitude_range)
            signal = signal * scale
        return signal
    
    def time_shift(self, signal):
        """Menggeser sinyal ke kiri/kanan"""
        if np.random.rand() < self.prob:
            shift = np.random.randint(-self.time_shift_max, self.time_shift_max)
            signal = torch.roll(signal, shifts=shift, dims=1)
        return signal

# ==========================================
# 3. DATASET UTAMA DENGAN SLIDING WINDOW
# ==========================================
class EEGDataset(Dataset):
    def __init__(self, data_dir, csv_path, fold, split, n_splits, window_sec, stride_sec, use_augmentation=False):
        self.data_dir = data_dir
        self.window_sec = window_sec
        self.stride_sec = stride_sec
        self.use_augmentation = use_augmentation
        
        # --- A. LOGIKA SPLITTING SUBJECT-WISE ---
        df = pd.read_csv(csv_path)
        
        # Pastikan kolom subject_id ada
        if 'subject_id' not in df.columns:
            df['subject_id'] = df['filename'].apply(lambda x: x.split('_')[0])

        # GroupKFold
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(df, df['label'], groups=df['subject_id']))
        train_idx, val_idx = splits[fold]
        
        # Pilih baris data sesuai split
        selected_idx = train_idx if split == 'train' else val_idx
        self.file_list = df.iloc[selected_idx][['filename', 'label']].values.tolist()
        
        # --- B. LOGIKA INDEXING SLIDING WINDOW ---
        self.samples = []
        print(f"[{split.upper()}] Mengindeks file dengan Sliding Window (Stride={stride_sec}s)...")
        
        for filename, label in self.file_list:
            file_path = os.path.join(self.data_dir, filename)
            
            if not os.path.exists(file_path):
                print(f"⚠️ File tidak ditemukan: {file_path}")
                continue
                
            try:
                raw_info = mne.io.read_raw_edf(file_path, preload=False, verbose='error')
                duration_sec = raw_info.times[-1]
                max_start = duration_sec - self.window_sec
                
                if max_start > 0:
                    starts = np.arange(0, max_start, self.stride_sec)
                    
                    # Simpan "Alamat" potongan data
                    for start in starts:
                        self.samples.append({
                            'file_path': file_path,
                            'start_sec': start,
                            'label': label
                        })
            except Exception as e:
                print(f"⚠️ Error membaca header {filename}: {e}")

        print(f"✅ Total {split.upper()} Samples Terkumpul: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # 1. Ambil info dari katalog
        sample_info = self.samples[idx]
        file_path = sample_info['file_path']
        start_sec = sample_info['start_sec']
        raw_label = sample_info['label']
        
        # --- [FIX] KONVERSI KSS (1-9) KE KELAS (0-2) ---
        if raw_label <= 3:
            label = 0   # Alert
        elif raw_label <= 6:
            label = 1   # Low Vigilance
        else:
            label = 2   # Drowsy
        # -----------------------------------------------
        
        try:
            # 2. Load File & Crop
            raw = mne.io.read_raw_edf(file_path, preload=True, verbose='error')
            
            # 3. Seleksi Channel
            target_channels = ['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H']
            
            # Fallback jika nama channel beda
            raw.pick_channels(target_channels)
            
            # 4. Potong Sinyal (CROP)
            t_end = start_sec + self.window_sec
            raw.crop(tmin=start_sec, tmax=t_end, include_tmax=False)
            
            # 5. Ambil Data (Microvolts)
            signal = raw.get_data() * 1e6  
            
            # 6. Z-Score Normalization
            mean = np.mean(signal, axis=1, keepdims=True)
            std = np.std(signal, axis=1, keepdims=True)
            signal = (signal - mean) / (std + 1e-6)
            
            # Convert ke Tensor PyTorch
            signal = torch.tensor(signal, dtype=torch.float32)
            
            # 7. Augmentasi
            if self.use_augmentation:
                aug = EEGAugmentation()
                signal = aug.add_gaussian_noise(signal)
                signal = aug.amplitude_scaling(signal)
                # Time shift optional
            
            return signal, torch.tensor(label, dtype=torch.long)

        except Exception as e:
            # Fallback jika file corrupt
            print(f"Error loading data idx {idx}: {e}")
            dummy_len = int(self.window_sec * Config.SAMPLE_RATE)
            return torch.zeros((Config.IN_CHANNELS, dummy_len)), torch.tensor(0, dtype=torch.long)

# ==========================================
# 4. FUNGSI COLLATE
# ==========================================
def collate_fn(batch):
    signals, labels = zip(*batch)
    signals = torch.stack(signals)
    labels = torch.stack(labels)
    return signals, labels

# ==========================================
# 5. BLOK TEST
# ==========================================
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