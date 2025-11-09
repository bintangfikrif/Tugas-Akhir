"""
Configuration file for Drowsiness Detection using Mamba
"""

class Config:
    # ==================== Model Architecture ====================
    MODEL_NAME = "mamba"  # or "resnet18" for baseline
    
    # Mamba specific parameters
    MAMBA_D_MODEL = 128  # Hidden dimension
    MAMBA_N_LAYERS = 4   # Number of Mamba layers
    MAMBA_D_STATE = 16   # SSM state expansion factor
    MAMBA_D_CONV = 4     # Local convolution width
    MAMBA_EXPAND = 2     # Block expansion factor
    
    # ==================== Data Parameters ====================
    IN_CHANNELS = 7      # EEG channels: Fz, Cz, C3, C4, Pz, EOG-V, EOG-H
    NUM_CLASSES = 9      # KSS levels: 1-9
    WINDOW_SEC = 5       # Window size in seconds
    SAMPLE_RATE = 512    # Hz
    
    # ==================== Training Parameters ====================
    BATCH_SIZE = 16
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    
    # Learning rate scheduler
    USE_SCHEDULER = True
    SCHEDULER_PATIENCE = 5
    SCHEDULER_FACTOR = 0.5
    
    # Early stopping
    EARLY_STOPPING_PATIENCE = 10
    
    # ==================== Cross Validation ====================
    N_SPLITS = 5         # 5-fold cross validation
    CURRENT_FOLD = 0
    
    # ==================== Loss Function ====================
    LOSS_TYPE = "ordinal"  # "ordinal" or "ce" (cross-entropy)
    ORDINAL_IMPORTANCE = 1.0  # Weight for ordinal loss
    
    # ==================== Data Augmentation ====================
    USE_AUGMENTATION = True
    AUG_GAUSSIAN_NOISE_STD = 0.01
    AUG_AMPLITUDE_SCALE_RANGE = (0.9, 1.1)
    AUG_TIME_SHIFT_MAX = 256  # samples
    
    # ==================== Paths ====================
    DATA_DIR = "psg"
    CHECKPOINT_DIR = "checkpoints"
    MLFLOW_TRACKING_URI = "./mlruns"
    MLFLOW_EXPERIMENT_NAME = "drowsiness-detection-mamba"
    
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
