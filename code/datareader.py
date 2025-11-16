import numpy as np
import os
from torch.utils.data import Dataset
import torch
import random
import pandas as pd
import mne
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from collections import Counter

# reproducibility
RANDOM_SEED = 23
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# KSS 1-9 to 3-class mapping
def kss_to_class(kss_level):
    if 1 <= kss_level <= 3:
        return 0  # Alert
    elif 4 <= kss_level <= 6:
        return 1  # Low Vigilance
    elif 7 <= kss_level <= 9:
        return 2  # Drowsy
    else:
        raise ValueError(f"Invalid KSS level: {kss_level}")

# compute class weights for imbalanced dataset
def compute_class_weights(labels, num_classes=3):
    label_counts = Counter(labels)
    total = sum(label_counts.values())
    
    weights = []
    for cls in range(num_classes):
        count = label_counts.get(cls, 1)  # avoid division by zero
        weight = total / (num_classes * count)
        weights.append(weight)
    
    return torch.FloatTensor(weights)

# EEG Dataset class
class EEGDataset(Dataset):
    TARGET_CHANNELS = ['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H']
    CLASS_NAMES = ['Alert', 'Low Vigilance', 'Drowsy']

    def __init__(self,
                 data_dir='psg',
                 fold=0,
                 split='train',
                 n_splits=5,
                 window_sec=1,  
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
        if not os.path.exists(csv_path):
            repo_root = os.path.dirname(os.path.dirname(__file__))
            alt = os.path.join(repo_root, 'label', 'labels.csv')
            if os.path.exists(alt):
                csv_path = alt

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"labels.csv not found at {csv_path}")

        df = pd.read_csv(csv_path)
        
        # Map KSS 1-9 to 3-class labels using kss_to_class function
        mapped = []
        for _, row in df.iterrows():
            fname = row['filename']
            lab = row['label'] 
            
            try:
                kss_level = int(lab)
                
                # Validate KSS range (1-9)
                if 1 <= kss_level <= 9:
                    # Convert to 3-class: 0=Alert, 1=Low Vig, 2=Drowsy
                    class_label = kss_to_class(kss_level)
                    mapped.append((fname, class_label))
                else:
                    print(f"Warning: Invalid KSS level {kss_level} in {fname}, skipping")
                    mapped.append((fname, None))
                    
            except Exception as e:
                print(f"Warning: Cannot parse label '{lab}' in {fname}: {e}")
                mapped.append((fname, None))

        self.label_dict = dict(mapped)
        self.labels = [self.label_dict.get(f, None) for f in self.edf_files]

        all_data = list(zip(self.edf_files, self.labels))

        # SUBJECT-LEVEL SPLIT) 
        
        def extract_subject_id(filename):
            """Extract subject ID from filename.
            Example: 'subject01_pvt1.edf' → 'subject01'
            """
            if 'subject' in filename.lower():
                parts = filename.split('_')
                for part in parts:
                    if 'subject' in part.lower():
                        return part
            return filename.split('_')[0]
        
        # Group by subject
        subject_to_files = {}
        for edf_file, label in all_data:
            if label is None:
                continue
            subject_id = extract_subject_id(edf_file)
            if subject_id not in subject_to_files:
                subject_to_files[subject_id] = []
            subject_to_files[subject_id].append((edf_file, label))
        
        # Get sorted list of unique subjects
        unique_subjects = sorted(subject_to_files.keys())
        n_subjects = len(unique_subjects)
        
        print(f"\n{'='*80}")
        print(f"SUBJECT-LEVEL CROSS-VALIDATION")
        print(f"{'='*80}")
        print(f"Total unique subjects: {n_subjects}")
        print(f"Subjects: {unique_subjects}")
        
        # 5-fold split on SUBJECTS
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
        
        splits = list(kfold.split(unique_subjects))
        train_idx, val_idx = splits[fold]
        
        train_subjects = [unique_subjects[i] for i in train_idx]
        val_subjects = [unique_subjects[i] for i in val_idx]
        
        print(f"\nFold {fold}:")
        print(f"  Train subjects ({len(train_subjects)}): {train_subjects}")
        print(f"  Val subjects ({len(val_subjects)}): {val_subjects}")
        
        # Collect files based on split
        if split == 'train':
            selected_subjects = train_subjects
        else:
            selected_subjects = val_subjects
        
        selected_files = []
        for subject in selected_subjects:
            selected_files.extend(subject_to_files[subject])
        
        print(f"  {split.capitalize()} files: {len(selected_files)}")
        print(f"{'='*80}\n")
        
        # generate list window untuk semua file dalam split
        self.sample_rate = 512  
        self.sample_len = self.window_sec * self.sample_rate 
        self.windows = []
        
        for edf_path, label in selected_files:
            full_path = os.path.join(self.data_dir, edf_path)
            try:
                raw = mne.io.read_raw_edf(full_path, preload=False, verbose=False)
                n_samples = raw.n_times
                max_start = n_samples - self.sample_len
                if max_start < 0:
                    continue  
                
                # buat daftar window start index
                for start in range(0, max_start, self.sample_len):
                    self.windows.append((edf_path, label, start))
                    
            except Exception as e:
                print(f"Error reading {edf_path}: {e}")
                continue
        
        random.shuffle(self.windows)
        
        # Print distribution statistics
        if len(self.windows) > 0:
            label_counts = Counter()
            for _, label, _ in self.windows:
                label_counts[label] += 1
            
            print(f"Total windows: {len(self.windows)}")
            print(f"Class distribution ({split}):")
            for cls in range(3):
                count = label_counts[cls]
                pct = 100 * count / len(self.windows)
                print(f"  {self.CLASS_NAMES[cls]:15s} (class {cls}): {count:5d} windows ({pct:5.1f}%)")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        edf_path, label, start = self.windows[idx]
        full_path = os.path.join(self.data_dir, edf_path)
        
        # Random offset untuk augmentasi (hanya di train)
        if self.random_offset and self.split == 'train':
            # offset max ± 0.25s (128 samples)
            max_offset = int(0.25 * self.sample_rate)
            offset = random.randint(-max_offset, max_offset)
            start = max(0, start + offset)
        
        # Baca hanya window yang dibutuhkan 
        try:
            raw = mne.io.read_raw_edf(full_path, preload=False, verbose=False)
            
            # Ensure start + sample_len doesn't exceed file length
            if start + self.sample_len > raw.n_times:
                start = raw.n_times - self.sample_len
            
            # Load specific time window
            start_sec = start / self.sample_rate
            stop_sec = (start + self.sample_len) / self.sample_rate
            raw_crop = raw.copy().crop(tmin=start_sec, tmax=stop_sec, include_tmax=False)
            raw_crop.load_data()
            
            # Pick channels
            raw_crop.pick_channels(self.TARGET_CHANNELS, ordered=True)
            
            # Get data: shape (n_channels, n_times)
            signals = raw_crop.get_data()  # (7, 512) for 1s window
            
            # Per-channel z-score normalization
            signals = (signals - signals.mean(axis=1, keepdims=True)) / (signals.std(axis=1, keepdims=True) + 1e-6)
            
            # Convert to tensor
            signals = torch.from_numpy(signals).float()
            
            # Apply transform if any
            if self.transform:
                signals = self.transform(signals)
            
            return signals, label
            
        except Exception as e:
            print(f"Error loading window from {edf_path}: {e}")
            # Return zeros if error
            return torch.zeros(len(self.TARGET_CHANNELS), self.sample_len), label


def get_dataloaders(data_dir='psg', fold=0, batch_size=32, num_workers=4):
    """
    Create train and val dataloaders with subject-level split.
    
    Returns:
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        class_weights: torch.FloatTensor of class weights for loss function
    """
    train_dataset = EEGDataset(
        data_dir=data_dir,
        fold=fold,
        split='train',
        window_sec=1,  # 1 second windows
        random_offset=True
    )
    
    val_dataset = EEGDataset(
        data_dir=data_dir,
        fold=fold,
        split='val',
        window_sec=1,  # 1 second windows
        random_offset=False
    )
    
    # Compute class weights from training set
    train_labels = [label for _, label, _ in train_dataset.windows]
    class_weights = compute_class_weights(train_labels, num_classes=3)
    
    print(f"\n{'='*80}")
    print(f"CLASS WEIGHTS (for handling imbalance)")
    print(f"{'='*80}")
    for cls in range(3):
        print(f"  {train_dataset.CLASS_NAMES[cls]:15s} (class {cls}): {class_weights[cls]:.3f}")
    print(f"{'='*80}\n")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Train windows: {len(train_dataset)}")
    print(f"Val windows: {len(val_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}\n")
    
    return train_loader, val_loader, class_weights


if __name__ == "__main__":
    print("Testing EEG Dataset with 3-class classification and 1s windows...")
    
    # Test dataset creation
    dataset = EEGDataset(
        data_dir='psg',
        fold=0,
        split='train',
        window_sec=1
    )
    
    if len(dataset) > 0:
        print(f"\nDataset size: {len(dataset)} windows")
        
        # Test a sample
        for idx in range(min(3, len(dataset))):
            print(f"\nSample window index {idx}")
            signals, label = dataset[idx]
            print(f"Signals shape: {signals.shape}")  # Should be (7, 512)
            print(f"Label: {label} ({dataset.CLASS_NAMES[label]})")
            print(f"Signal range: [{signals.min():.3f}, {signals.max():.3f}]")
    else:
        print("No data found. Check your data_dir path.")
    
    # Test dataloader
    print("\n" + "="*80)
    print("Testing get_dataloaders()...")
    print("="*80)
    
    try:
        train_loader, val_loader, class_weights = get_dataloaders(
            data_dir='psg',
            fold=0,
            batch_size=16,
            num_workers=0
        )
        
        print(f"\nClass weights: {class_weights}")
        
        # Test one batch
        for batch_signals, batch_labels in train_loader:
            print(f"\nBatch signals shape: {batch_signals.shape}")  # (B, 7, 512)
            print(f"Batch labels shape: {batch_labels.shape}")  # (B,)
            print(f"Batch labels: {batch_labels}")
            break
        
        print("\n✅ Datareader test successful!")
        
    except Exception as e:
        print(f"\n❌ Error testing dataloader: {e}")
        import traceback
        traceback.print_exc()