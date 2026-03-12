"""
Configuration file for Drowsiness Detection using Mamba
"""

class Config:
    # ==================== Model Architecture ====================
    MODEL_NAME = "mamba" 
    
    # Mamba specific parameters
    MAMBA_D_MODEL = 32   # Naikkan kapasitas (dari 32)
    MAMBA_N_LAYERS = 4   # Number of Mamba layers
    MAMBA_D_STATE = 16   # SSM state expansion factor
    MAMBA_D_CONV = 4     # Local convolution width
    MAMBA_EXPAND = 2     # Block expansion factor
    
    # ==================== Data Parameters ====================
    IN_CHANNELS = 7      # EEG channels: Fz, Cz, C3, C4, Pz, EOG-V, EOG-H
    NUM_CLASSES = 3      # 2-class: Alert(0) vs Drowsy(1)
    WINDOW_SEC = 30       # Window size in seconds
    SAMPLE_RATE = 512    # Hz
    STRIDE_SEC = 10      # Sliding window stride untuk training (detik)
    
    # ==================== Training Parameters ====================
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-4  
    WEIGHT_DECAY = 1e-3
    
    # Learning rate scheduler
    USE_SCHEDULER = True
    SCHEDULER_PATIENCE = 5   # Naikkan dari 3: beri waktu lebih sebelum reduce LR
    SCHEDULER_FACTOR = 0.5
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 15  # Naikkan dari 10
    
    # ==================== Cross Validation ====================
    N_SPLITS = 14         # 5-fold cross validation
    CURRENT_FOLD = 0
    
    # ==================== Loss Function ====================
    LOSS_TYPE = "ordinal"  # "ordinal" or "ce" (cross-entropy)
    ORDINAL_IMPORTANCE = 1.0  # Weight for ordinal loss
    
    # ==================== Data Augmentation ====================
    USE_BANDPASS_FILTER = False
    USE_AUGMENTATION = True
    AUG_GAUSSIAN_NOISE_STD = 0.01
    AUG_AMPLITUDE_SCALE_RANGE = (0.9, 1.1)
    AUG_TIME_SHIFT_MAX = 256  # samples
    
    # ==================== Paths ====================
    DATA_DIR = "psg"
    CHECKPOINT_DIR = "checkpoints"
    USE_WANDB = True
    WANDB_PROJECT = "Drowsiness-Mamba-DROZY"
    
    # ==================== Device ====================
    USE_CUDA = True
    NUM_WORKERS = 2  # For DataLoader
    
    # ==================== Logging ====================
    LOG_INTERVAL = 10  # Log every N batches
    SAVE_BEST_ONLY = True
    
    @classmethod
    def to_dict(cls):
        """Convert config to dictionary for MLflow logging"""
        return {
            key: value for key, value in cls.__dict__.items()
            if not key.startswith('__') and not callable(value)
        }