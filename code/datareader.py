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
    """
    EEG Dataset for 9-class KSS ordinal regression (KSS levels 1-9)
    
    Returns:
        signals: (C, T) tensor of EEG/EOG signals
        label: int from 0-8 (representing KSS 1-9)
        ordinal_labels: (8,) binary tensor for ordinal regression
    """
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
        
        # Map KSS 1-9 to class labels 0-8
        # KSS level 1 -> class 0, KSS level 2 -> class 1, ..., KSS level 9 -> class 8
        mapped = []
        for _, row in df.iterrows():
            fname = row['filename']
            lab = row['label'] 
            
            try:
                kss_level = int(lab)
                
                # Validate KSS range (1-9)
                if 1 <= kss_level <= 9:
                    # Convert to 0-indexed: KSS 1-9 -> class 0-8
                    class_label = kss_level - 1
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
                    print(f"Warning: Skipping {fname} due to missing channels.")
                    continue
                    
                n_samples = raw.n_times
            except Exception as e:
                print(f"Error reading {fname}: {e}. Skipping file.")
                continue

            # random offset
            offset = random.randint(0, self.sample_rate) if self.random_offset else 0

            # buat daftar window start index
            for start in range(offset, n_samples - self.sample_len, self.sample_len):
                self.windows.append((edf_path, label, start))

        random.shuffle(self.windows)
        
        # Print dataset statistics
        if len(self.windows) > 0:
            label_counts = {}
            for _, label, _ in self.windows:
                label_counts[label] = label_counts.get(label, 0) + 1
            print(f"\n{split.upper()} Dataset Statistics (Fold {fold}):")
            print(f"Total windows: {len(self.windows)}")
            print("Class distribution (KSS level -> count):")
            for kss_class in sorted(label_counts.keys()):
                kss_level = kss_class + 1  # Convert back to KSS 1-9
                print(f"  KSS {kss_level}: {label_counts[kss_class]} windows")

    def __len__(self):
        return len(self.windows)

    def _create_ordinal_labels(self, class_label):
        """
        Create ordinal labels for ordinal regression.
        For 9 classes (0-8), we need 8 binary classifiers.
        
        Example: if class_label = 5 (KSS level 6):
        ordinal_labels = [1, 1, 1, 1, 1, 0, 0, 0]
        Meaning: P(Y > 0) = 1, P(Y > 1) = 1, ..., P(Y > 5) = 0
        
        Args:
            class_label: int from 0 to 8
        
        Returns:
            ordinal_labels: (8,) binary tensor
        """
        ordinal_labels = torch.zeros(8, dtype=torch.float32)
        for i in range(class_label):
            ordinal_labels[i] = 1.0
        return ordinal_labels

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
                raise RuntimeError(f"Channel {target_ch} not found in {edf_path} at __getitem__")

        signals = np.stack(reordered_data)
        signals_tensor = torch.tensor(signals, dtype=torch.float32)

        if self.transform:
            signals_tensor = self.transform(signals_tensor)

        # Create ordinal labels for ordinal regression
        ordinal_labels = self._create_ordinal_labels(label)

        return signals_tensor, label, ordinal_labels


def default_transform(x: torch.Tensor) -> torch.Tensor:
    """
    Per-channel z-score normalization.
    x shape: (channels, samples)
    """
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)
    return (x - mean) / (std + 1e-6)


def collate_fn(batch):
    """
    Custom collate function for DataLoader
    
    Returns:
        signals: (B, C, T)
        labels: (B,) class labels (0-8)
        ordinal_labels: (B, 8) ordinal binary labels
    """
    signals = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    ordinal_labels = [item[2] for item in batch]
    
    signals = torch.stack(signals, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    ordinal_labels = torch.stack(ordinal_labels, dim=0)
    
    return signals, labels, ordinal_labels


if __name__ == "__main__":
    data_dir_path = 'psg' 
    
    if not os.path.exists(data_dir_path):
        print(f"Directory '{data_dir_path}' not found.")
        print("Make sure you run this script from the correct directory,")
        print("or change the 'data_dir_path' variable in `if __name__ == '__main__':`")
    else:
        fold = 0  
        train_dataset = EEGDataset(
            data_dir=data_dir_path, 
            split='train', 
            fold=fold,
            transform=default_transform
        )
        val_dataset = EEGDataset(
            data_dir=data_dir_path, 
            split='val', 
            fold=fold,
            transform=default_transform
        )

        print(f"\n{'='*60}")
        print(f"Fold {fold+1} Summary")
        print(f"{'='*60}")
        print(f"Train windows: {len(train_dataset)}")
        print(f"Val windows: {len(val_dataset)}")

        # Test get item
        if len(train_dataset) > 0:
            idx = random.randint(0, len(train_dataset) - 1)
            signals, label, ordinal_labels = train_dataset[idx]

            print(f"\n{'='*60}")
            print(f"Sample window index {idx}")
            print(f"{'='*60}")
            print(f"Signals shape: {signals.shape}")  # (7, 2560)
            print(f"Class label: {label} (KSS level {label + 1})")
            print(f"Ordinal labels: {ordinal_labels}")
            print(f"Channel names: {train_dataset.TARGET_CHANNELS}")

            # Plot signals
            n_channels = signals.shape[0]
            fig, axes = plt.subplots(n_channels, 1, figsize=(14, 10), sharex=True)
            kss_level = label + 1
            fig.suptitle(f"KSS Level {kss_level} (Class {label})", fontsize=16, fontweight='bold')

            for i in range(n_channels):
                axes[i].plot(signals[i].numpy(), linewidth=0.5)
                axes[i].set_ylabel(train_dataset.TARGET_CHANNELS[i], 
                                 rotation=0, labelpad=40, fontsize=10, fontweight='bold')
                axes[i].grid(True, alpha=0.3)
                axes[i].tick_params(labelsize=8)

            axes[-1].set_xlabel("Sample Points", fontsize=10)
            plt.tight_layout(rect=[0, 0, 1, 0.97])
            
            # Save plot
            os.makedirs('outputs', exist_ok=True)
            plt.savefig(f'outputs/sample_kss_{kss_level}.png', dpi=150, bbox_inches='tight')
            print(f"\nPlot saved to: outputs/sample_kss_{kss_level}.png")
            plt.close()
        else:
            print("\nNo training data to plot. Check data folder and CSV file.")
