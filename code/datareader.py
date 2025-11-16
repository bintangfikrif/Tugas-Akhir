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

# UTILITY FUNCTIONS FOR DATA PREPROCESSING

def default_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Per-channel Z-score normalization.
    
    Args:
        x: Input tensor of shape (channels, samples)
        
    Returns:
        Normalized tensor with mean~0 and std~1 per channel
        
    Example:
        >>> signals = torch.randn(7, 512)  # 7 channels, 512 samples
        >>> normalized = default_transform(signals)
        >>> normalized.mean(dim=1)  # Should be close to 0 for each channel
    """
    # Compute per-channel mean and std
    mean = x.mean(dim=1, keepdim=True)  # Shape: (channels, 1)
    std = x.std(dim=1, keepdim=True)    # Shape: (channels, 1)
    
    # Normalize: (x - mean) / std
    # Add small epsilon to prevent division by zero
    return (x - mean) / (std + 1e-6)


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    
    Handles batching of EEG samples and ensures labels are properly formatted.
    
    Args:
        batch: List of tuples from Dataset.__getitem__
               Each tuple: (signals_tensor, label, filepath, ch_names, start)
               
    Returns:
        signals: Tensor of shape (batch_size, channels, samples)
        labels: Tensor of shape (batch_size,) with dtype long
        
    Example:
        >>> # In DataLoader
        >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    """
    # Extract signals and labels from batch
    signals = [item[0] for item in batch]  # List of (channels, samples) tensors
    labels = [item[1] for item in batch]   # List of integer labels
    
    # Stack signals into batch dimension
    signals = torch.stack(signals, dim=0)  # Shape: (batch_size, channels, samples)
    
    # Convert labels to tensor, handling None values
    labels = torch.tensor(
        [int(l) if l is not None else 0 for l in labels], 
        dtype=torch.long
    )
    
    return signals, labels


# DATASET CLASS

class EEGDataset(Dataset):
    """
    EEG Dataset for 3-class drowsiness detection.
    
    Classes:
        0: Alert (KSS 1-3)
        1: Low Vigilance (KSS 4-6)
        2: Drowsy (KSS 7-9)
    """
    TARGET_CHANNELS = ['Fz', 'Cz', 'C3', 'C4', 'Pz', 'EOG-V', 'EOG-H']

    def __init__(self,
                 data_dir='psg',
                 fold=0,
                 split='train',
                 n_splits=5,
                 window_sec=1,  # Changed default to 1 second
                 transform=None,
                 random_offset=True):
        """
        Initialize EEG Dataset.
        
        Args:
            data_dir: Directory containing EDF files
            fold: Fold index for cross-validation (0 to n_splits-1)
            split: 'train' or 'val'
            n_splits: Number of folds for cross-validation
            window_sec: Window size in seconds
            transform: Transform function to apply to signals
            random_offset: Whether to use random offset for window extraction
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        self.window_sec = window_sec
        self.random_offset = random_offset

        # List EDF files
        self.edf_files = [f for f in os.listdir(data_dir) if f.endswith('.edf')]
        self.edf_files.sort()

        # Read label CSV
        csv_path = os.path.join(os.path.dirname(data_dir), 'label', 'labels.csv')
        if not os.path.exists(csv_path):
            # Fallback: label folder expected as sibling of this script's parent
            repo_root = os.path.dirname(os.path.dirname(__file__))
            alt = os.path.join(repo_root, 'label', 'labels.csv')
            if os.path.exists(alt):
                csv_path = alt

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"labels.csv not found at {csv_path}")

        df = pd.read_csv(csv_path)
        
        # Map KSS levels to 3 classes
        # KSS 1-3 → Class 0 (Alert)
        # KSS 4-6 → Class 1 (Low Vigilance)
        # KSS 7-9 → Class 2 (Drowsy)
        mapped = []
        for _, row in df.iterrows():
            fname = row['filename']
            kss = row['label']
            
            try:
                kss_level = int(kss)
                
                if 1 <= kss_level <= 3:
                    class_label = 0  # Alert
                elif 4 <= kss_level <= 6:
                    class_label = 1  # Low Vigilance
                elif 7 <= kss_level <= 9:
                    class_label = 2  # Drowsy
                else:
                    class_label = None  # Invalid KSS
                
                mapped.append((fname, class_label))
                    
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

        # Create windows
        for fname, label in self.data:
            if label is None:
                continue
                
            edf_path = os.path.join(self.data_dir, fname)
            try:
                raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
                if not all(ch in raw.ch_names for ch in self.TARGET_CHANNELS):
                    print(f"Warning: Skipping {fname} due to missing channels.")
                    continue
                    
                n_samples = raw.n_times
            except Exception as e:
                print(f"Error reading {fname}: {e}. Skipping file.")
                continue

            # Random offset
            offset = random.randint(0, self.sample_rate) if self.random_offset else 0

            # Create window start indices
            for start in range(offset, n_samples - self.sample_len, self.sample_len):
                self.windows.append((edf_path, label, start))

        random.shuffle(self.windows)
        
        # Print dataset statistics
        if len(self.windows) > 0:
            label_counts = {}
            for _, label, _ in self.windows:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            print(f"\n{split.upper()} Dataset Statistics (Fold {fold}):")
            print(f"  Total windows: {len(self.windows)}")
            print(f"  Class 0 (Alert): {label_counts.get(0, 0)} windows")
            print(f"  Class 1 (Low Vigilance): {label_counts.get(1, 0)} windows")
            print(f"  Class 2 (Drowsy): {label_counts.get(2, 0)} windows")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        edf_path, label, start = self.windows[idx]

        # Load EDF
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
                raise RuntimeError(f"Channel {target_ch} not found in {edf_path} at __getitem__")

        signals = np.stack(reordered_data)
        signals_tensor = torch.tensor(signals, dtype=torch.float32)

        if self.transform:
            signals_tensor = self.transform(signals_tensor)

        return signals_tensor, label, edf_path, self.TARGET_CHANNELS, start


if __name__ == "__main__":
    data_dir_path = 'psg' 
    
    if not os.path.exists(data_dir_path):
        print(f"Data directory '{data_dir_path}' not found.")
        print("Make sure you run this script from the correct directory,")
        print("or update the 'data_dir_path' variable in `if __name__ == '__main__':`")
    else:
        fold = 0  
        train_dataset = EEGDataset(data_dir=data_dir_path, split='train', fold=fold, window_sec=1)
        val_dataset = EEGDataset(data_dir=data_dir_path, split='val', fold=fold, window_sec=1)

        print(f"\nFold {fold+1}")
        print(f"Train windows: {len(train_dataset)}")
        print(f"Val windows: {len(val_dataset)}")

        # Test one sample and plot
        if len(train_dataset) > 0:
            idx = random.randint(0, len(train_dataset) - 1)
            signals, label, filepath, ch_names, start = train_dataset[idx]

            print(f"\nExample window index {idx}")
            print(f"Signals shape: {signals.shape}")  # (7, 512) for 1 second window
            print(f"Label: {label}")  # Should be 0, 1, or 2
            
            class_names = {0: 'Alert (KSS 1-3)', 1: 'Low Vigilance (KSS 4-6)', 2: 'Drowsy (KSS 7-9)'}
            print(f"Class: {class_names[label]}")
            print(f"File path: {filepath}")
            print(f"Start index: {start}")
            print(f"Channel names: {ch_names}")

            # Plot signals
            n_channels = signals.shape[0]
            fig, axes = plt.subplots(n_channels, 1, figsize=(12, 10), sharex=True)
            title = f"Class {label}: {class_names[label]}"
            fig.suptitle(f"Signals from {os.path.basename(filepath)} | {title} | Start: {start}", fontsize=14)

            for i in range(n_channels):
                axes[i].plot(signals[i].numpy())
                axes[i].set_ylabel(ch_names[i], rotation=0, labelpad=30)
                axes[i].grid(True)

            axes[-1].set_xlabel("Sample Points")
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
        else:
            print("\nNo training data to plot. Check your data folder and CSV file.")