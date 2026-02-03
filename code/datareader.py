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


class EEGAugmentation:
    """
    Augmentasi data EEG untuk meningkatkan generalisasi model.
    Diterapkan hanya pada data training.
    
    Metode augmentasi:
    1. Gaussian Noise - Menambahkan noise acak
    2. Amplitude Scaling - Mengubah skala amplitudo
    3. Time Shift - Menggeser sinyal secara temporal
    
    Ref: Sesuai dengan Config.py yang telah didefinisikan
    """
    def __init__(self, 
                 gaussian_std=0.01,
                 amplitude_range=(0.9, 1.1),
                 time_shift_max=256,
                 prob=0.5,
                 use_augmentation=True):
        """
        Args:
            gaussian_std: Standar deviasi untuk Gaussian noise
            amplitude_range: Range untuk amplitude scaling (min, max)
            time_shift_max: Maksimal pergeseran waktu (dalam samples)
            prob: Probabilitas untuk setiap jenis augmentasi
            use_augmentation: Flag untuk enable/disable augmentasi
        """
        self.gaussian_std = gaussian_std
        self.amplitude_range = amplitude_range
        self.time_shift_max = time_shift_max
        self.prob = prob
        self.use_augmentation = use_augmentation
    
    def add_gaussian_noise(self, signal):
        """
        Menambahkan Gaussian noise ke sinyal.
        Berguna untuk membuat model lebih robust terhadap noise.
        
        Args:
            signal: (C, T) tensor
        Returns:
            Augmented signal
        """
        if np.random.rand() < self.prob:
            noise = torch.randn_like(signal) * self.gaussian_std
            signal = signal + noise
        return signal
    
    def amplitude_scaling(self, signal):
        """
        Mengubah skala amplitudo sinyal secara random.
        Mensimulasikan variasi intensitas sinyal antar subjek.
        
        Args:
            signal: (C, T) tensor
        Returns:
            Augmented signal
        """
        if np.random.rand() < self.prob:
            scale = np.random.uniform(*self.amplitude_range)
            signal = signal * scale
        return signal
    
    def time_shift(self, signal):
        """
        Menggeser sinyal secara temporal (circular shift).
        Membantu model belajar invariansi terhadap translasi temporal.
        
        Args:
            signal: (C, T) tensor
        Returns:
            Augmented signal
        """
        if np.random.rand() < self.prob:
            shift = np.random.randint(-self.time_shift_max, self.time_shift_max)
            signal = torch.roll(signal, shift, dims=-1)
        return signal
    
    def __call__(self, signal):
        """
        Aplikasikan augmentasi secara berurutan.
        
        Args:
            signal: (C, T) tensor - 7 channels × time points
        Returns:
            Augmented signal
        """
        if not self.use_augmentation:
            return signal
        
        # Terapkan augmentasi secara berurutan
        signal = self.add_gaussian_noise(signal)
        signal = self.amplitude_scaling(signal)
        signal = self.time_shift(signal)
        
        return signal


class EEGDataset(Dataset):
    """
    Dataset DROZY untuk Klasifikasi 3-Kelas Kantuk.
    
    Features:
    - Window: 30 Detik sesuai standar AASM
    - Validasi: Subject-Wise K-Fold Cross Validation
    - Normalisasi: Z-Score per subjek
    - Augmentasi: Gaussian Noise, Amplitude Scaling, Time Shift (hanya training)
    
    Kelas:
    - 0: Alert (KSS 1-3)
    - 1: Low Vigilance (KSS 4-6)
    - 2: Drowsy (KSS 7-9)
    """
    TARGET_CHANNELS = ['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H']

    def __init__(self,
                 data_dir='psg',
                 csv_path='label/labels.csv',
                 fold=0,
                 split='train',
                 n_splits=5,
                 window_sec=30,
                 # ✅ Parameter Augmentasi
                 use_augmentation=True,
                 aug_gaussian_std=0.01,
                 aug_amplitude_range=(0.9, 1.1),
                 aug_time_shift_max=256,
                 aug_prob=0.5):
        
        self.data_dir = data_dir
        self.window_sec = window_sec
        self.sample_rate = 512
        self.sample_len = window_sec * self.sample_rate  # 30 * 512 = 15360
        self.split = split
        
        # ✅ Inisialisasi Augmentasi (hanya untuk training)
        if split == 'train' and use_augmentation:
            self.transform = EEGAugmentation(
                gaussian_std=aug_gaussian_std,
                amplitude_range=aug_amplitude_range,
                time_shift_max=aug_time_shift_max,
                prob=aug_prob,
                use_augmentation=True
            )
            print(f"✅ Data Augmentation ENABLED untuk {split} set")
            print(f"   ├─ Gaussian Noise: std={aug_gaussian_std}")
            print(f"   ├─ Amplitude Scaling: {aug_amplitude_range}")
            print(f"   ├─ Time Shift: max={aug_time_shift_max} samples")
            print(f"   └─ Probability: {aug_prob}")
        else:
            self.transform = None
            if split == 'train':
                print(f"⚠️  Data Augmentation DISABLED untuk {split} set")

        # 1. Load Labels & Mapping KSS ke 3 Kelas
        df = pd.read_csv(csv_path)
        self.label_dict = {}
        
        for _, row in df.iterrows():
            kss = int(row['label'])
            # Mapping 3 Kelas: Alert(0), Low Vigilance(1), Drowsy(2)
            # Sesuai proposal hal. 32
            if kss <= 3:
                label = 0  # Alert
            elif kss <= 6:
                label = 1  # Low Vigilance
            else:
                label = 2  # Drowsy
            
            self.label_dict[row['filename']] = label

        # 2. Subject-Wise Split untuk mencegah data leakage
        self.edf_files = [
            f for f in os.listdir(data_dir) 
            if f.endswith('.edf') and f in self.label_dict
        ]
        self.edf_files.sort()
        
        # Extract subject ID dari filename (format: SubjectID-SessionID.edf)
        self.subjects = [f.split('-')[0] for f in self.edf_files]

        # Group K-Fold untuk memastikan satu subjek hanya di satu fold
        gkf = GroupKFold(n_splits=n_splits)
        folds = list(gkf.split(self.edf_files, groups=self.subjects))
        
        train_idx, val_idx = folds[fold]
        selected_files = [
            self.edf_files[i] for i in (train_idx if split == 'train' else val_idx)
        ]

        print(f"\n📂 Dataset Info - Fold {fold} ({split}):")
        print(f"   ├─ Total files: {len(selected_files)}")
        print(f"   ├─ Unique subjects: {len(set([f.split('-')[0] for f in selected_files]))}")

        # 3. Pre-compute statistik normalisasi per subjek
        # Penting untuk konsistensi normalisasi Z-score
        print(f"   └─ Computing normalization statistics...")
        self.subject_stats = self._compute_subject_stats(selected_files)

        # 4. Pre-load semua windows untuk efisiensi
        self.windows = []
        class_counts = {0: 0, 1: 0, 2: 0}
        
        for fname in selected_files:
            label = self.label_dict[fname]
            edf_path = os.path.join(self.data_dir, fname)
            
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                
                # Buat jendela 30 detik tanpa overlap
                num_windows = (raw.n_times - self.sample_len) // self.sample_len
                
                for i in range(num_windows):
                    start = i * self.sample_len
                    self.windows.append((fname, edf_path, label, start))
                    class_counts[label] += 1
                    
            except Exception as e:
                print(f"⚠️  Error loading {fname}: {e}")
        
        # Print distribusi kelas
        print(f"\n📊 Class Distribution ({split}):")
        print(f"   ├─ Alert (0): {class_counts[0]} samples")
        print(f"   ├─ Low Vigilance (1): {class_counts[1]} samples")
        print(f"   └─ Drowsy (2): {class_counts[2]} samples")
        print(f"   Total: {len(self.windows)} windows\n")

    def _compute_subject_stats(self, files):
        """
        Menghitung mean & std per subjek untuk normalisasi yang konsisten.
        Sesuai strategi normalisasi Z-score berbasis data latih (proposal hal. 33).
        
        PENTING: Normalisasi harus dilakukan per subjek, bukan per window,
        untuk menghindari kebocoran informasi.
        
        Args:
            files: List of EDF filenames
        
        Returns:
            dict: Dictionary berisi statistik per file
        """
        stats = {}
        
        for fname in files:
            edf_path = os.path.join(self.data_dir, fname)
            
            try:
                # Load seluruh data subjek
                raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
                raw.pick_channels(self.TARGET_CHANNELS, ordered=True)
                data, _ = raw[:]  # (C, T) - All channels, all time points
                
                # Hitung statistik global subjek
                stats[fname] = {
                    'mean': float(data.mean()),
                    'std': float(data.std())
                }
                
            except Exception as e:
                print(f"⚠️  Error computing stats for {fname}: {e}")
                # Fallback ke nilai default
                stats[fname] = {'mean': 0.0, 'std': 1.0}
        
        return stats

    def _normalize_subject(self, tensor, fname):
        """
        Z-Score Normalization menggunakan statistik subjek.
        Z = (X - mu) / sigma
        
        Sesuai Rumus 2.1 (hal. 15) Proposal Tugas Akhir.
        
        Args:
            tensor: (C, T) EEG signal
            fname: Filename untuk lookup statistik
        
        Returns:
            Normalized tensor
        """
        mu = self.subject_stats[fname]['mean']
        sigma = self.subject_stats[fname]['std']
        
        # Normalisasi dengan epsilon untuk stabilitas numerik
        normalized = (tensor - mu) / (sigma + 1e-6)
        
        return normalized

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        """
        Load dan preprocess satu window data.
        
        Pipeline:
        1. Load raw EEG/EOG signal
        2. Apply augmentation (hanya training)
        3. Apply normalization
        
        Returns:
            tuple: (tensor, label)
                - tensor: (7, 15360) - 7 channels × 30 seconds @ 512Hz
                - label: int (0, 1, atau 2)
        """
        fname, edf_path, label, start = self.windows[idx]
        
        # Load segmen data 30 detik
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        raw.pick_channels(self.TARGET_CHANNELS, ordered=True)
        
        # Extract window
        data, _ = raw[:, start : start + self.sample_len]
        
        # Convert ke PyTorch tensor
        tensor = torch.tensor(data, dtype=torch.float32)

        # ✅ 1. Terapkan Augmentasi (HANYA untuk training set)
        if self.transform is not None:
            tensor = self.transform(tensor)

        # ✅ 2. Terapkan Normalisasi (WAJIB untuk semua set)
        # Menggunakan statistik subjek yang sudah di-precompute
        tensor = self._normalize_subject(tensor, fname)

        return tensor, label
    

def collate_fn(batch):
    """
    Custom collate function untuk menggabungkan sampel EEG menjadi batch.
    Sesuai standar pemrosesan sinyal 30 detik.
    
    Args:
        batch: List of (signal, label) tuples
    
    Returns:
        tuple: (signals, labels)
            - signals: (B, 7, 15360) tensor
            - labels: (B,) tensor
    """
    signals = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    
    return signals, labels


# ✅ Test function untuk validasi dataset
def test_dataset():
    """
    Fungsi untuk testing dataset secara cepat.
    Jalankan dengan: python datareader.py
    """
    print("="*60)
    print("🧪 TESTING DATASET")
    print("="*60)
    
    # Test dengan augmentasi
    dataset_train = EEGDataset(
        data_dir='psg',
        csv_path='label/labels.csv',
        fold=0,
        split='train',
        n_splits=5,
        window_sec=30,
        use_augmentation=True  # Enable augmentation
    )
    
    # Test tanpa augmentasi
    dataset_val = EEGDataset(
        data_dir='psg',
        csv_path='label/labels.csv',
        fold=0,
        split='val',
        n_splits=5,
        window_sec=30,
        use_augmentation=False  # Disable for validation
    )
    
    print(f"\n✅ Training set size: {len(dataset_train)}")
    print(f"✅ Validation set size: {len(dataset_val)}")
    
    # Test loading satu sample
    signal, label = dataset_train[0]
    print(f"\n📊 Sample shape: {signal.shape}")
    print(f"📊 Sample label: {label}")
    print(f"📊 Signal range: [{signal.min():.4f}, {signal.max():.4f}]")
    print(f"📊 Signal mean: {signal.mean():.4f}")
    print(f"📊 Signal std: {signal.std():.4f}")
    
    # Test DataLoader
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset_train, batch_size=4, shuffle=True, collate_fn=collate_fn)
    
    batch_signals, batch_labels = next(iter(loader))
    print(f"\n📦 Batch signals shape: {batch_signals.shape}")
    print(f"📦 Batch labels shape: {batch_labels.shape}")
    print(f"📦 Batch labels: {batch_labels}")
    
    print("\n" + "="*60)
    print("✅ DATASET TEST PASSED!")
    print("="*60)


if __name__ == "__main__":
    test_dataset()