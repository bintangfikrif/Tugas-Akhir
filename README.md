# EEG-EOG Based Drowsiness Detection 

 **Tugas Akhir** - Deteksi tingkat kantuk pengemudi menggunakan State Space Model (Mamba) berbasis sinyal EEG dan EOG.

## Deskripsi Project

Project ini berfokus pada **deteksi tingkat kantuk pengemudi** dengan memanfaatkan sinyal biologis:
- **EEG (Electroencephalography)**: Sinyal aktivitas otak dari 5 channel (Fz, Cz, C3, C4, Pz)
- **EOG (Electrooculography)**: Sinyal gerakan mata dari 2 channel (EOG-V, EOG-H)

Model yang digunakan adalah **Mamba**, sebuah arsitektur berbasis State Space Model yang efisien untuk pemrosesan sequential data. Model ini melampaui performa Transformer untuk tugas-tugas yang melibatkan long-range dependencies dalam data temporal.

### Tujuan
Membangun sistem deteksi kantuk ringan yang akurat dengan:
- Preprocessing sinyal biomedis (filtering, normalisasi, windowing)
- Ekstraksi fitur temporal menggunakan Mamba
- Prediksi tingkat kantuk dalam skala ordinal (0-9, KSS - Karolinska Sleepiness Scale)

---

## Gambaran Umum Project

### Dataset
- **Sumber**: The ULg Multimodality Drowsiness Database (called DROZY) and Examples of Use. https://www.drozy.uliege.be/
- **Format**: Polysomnography (PSG) recordings dalam format EDF
- **Total Subjects**: 14 partisipan
- **Total Recordings**: 36 file EDF (variasi 1-3 recordings per subjek)
- **Durasi**: Data mentah dari electrodes sesuai standard PSG
- **Label**: KSS scores (1-9) yang merepresentasikan tingkat kantuk

Link Dataset: [DROZY Dataset](https://drive.google.com/drive/folders/1a7vL9jGtV7vvHAWTnrj3r_GN1jK4-DOU?usp=sharing)

### Preprocessing
- **Sampling Rate**: 512 Hz
- **Window**: 30 detik sliding window dengan stride 10 detik
- **Channels**: 7 channel (5 EEG + 2 EOG)
- **Normalisasi**: Z-score normalization per subjek

### Model Architecture
- **Backbone**: Mamba (State Space Model)
- **Layers**: 4 Mamba layers dengan dimension 32
- **Output**: Single output untuk regression task (KSS prediction)
- **Training**: 5-fold cross-validation

---

## Struktur Project

```
ta_if_mct/
├── code/                          # Source code utama
│   ├── baseline_model.py          # Model baseline untuk comparison
│   ├── config.py                  # Konfigurasi hyperparameter
│   ├── datareader.py              # Data loader dan preprocessing
│   ├── losses.py                  # Custom loss functions
│   ├── models.py                  # Definisi model Mamba
│   ├── train.py                   # Script training
│   ├── utils.py                   # Utility functions
│   ├── mamba_profiler.py          # Model profiling (FLOPs, parameters)
│   ├── checkpoints/               # Pre-trained models
│   │   ├── best_classification_run/
│   │   └── best_regression_run/
│   ├── label/
│   │   └── labels.csv             # Ground truth KSS labels
│   ├── psg/                       # Raw PSG data (EDF files)
│   │   ├── 1-1.edf, 1-2.edf, ...  # 36 file EDF
│   └── visualization/             # Scripts visualisasi
│       ├── visualize_augmentation.py
│       └── visualize_normalization.py
├── thesis/                        # LaTeX thesis files
│   ├── thesis.tex                 # Main thesis
│   ├── references.bib             # Bibliography
│   └── chapters/                  # Chapter files
├── LOGBOOK.md                     # Research logbook
├── README.md                      # File ini
└── requirements.txt               # Python dependencies
```

---

## System Requirements

### Hardware
- **Minimum**: CPU dengan 8GB RAM
- **Recommended**: GPU dengan CUDA support (NVIDIA) untuk training yang lebih cepat
- **Storage**: ~5-10 GB untuk raw data dan checkpoints


### Software
- **Python**: 3.8+
- **OS**: Windows, macOS, Linux

NOTE: Windows perlu WSL untuk menjalankan ```mamba-ssm``` 

### Library Dependencies
```
PyTorch 2.4.0 (torch, torchvision, torchaudio)
Mamba-SSM 2.2.2 (State Space Model)
NumPy, Pandas, Scikit-learn
MNE 1.x (EDF file reading dan EEG processing)
Matplotlib, Seaborn (Visualization)
Weights & Biases (wandb) - untuk experiment tracking
```

Lihat [requirements.txt](requirements.txt) untuk daftar lengkap dependencies.

---

## Instalasi dan Setup

### 1. Clone Repository
```bash
git clone https://github.com/bintangfikrif/Tugas-Akhir.git
cd Tugas-Akhir/ta_if_mct
```

### 2. Setup Virtual Environment
**Windows (PowerShell):**
```powershell
# Create virtual environment
python -m venv ta

# Activate
.\ta\Scripts\Activate.ps1
```

**macOS/Linux:**
```bash
python3 -m venv ta
source ta/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Verify Installation
```bash
python -c "import torch; print(torch.__version__)"
python -c "from mamba_ssm import Mamba; print('Mamba OK')"
python -c "import mne; print('MNE OK')"
```

---

## Dataset Setup

Dataset sudah tersedia dalam folder `code/psg/` dan `code/label/`:

```
code/
├── psg/              # 39 EDF files (raw PSG data)
├── label/
│   └── labels.csv    # CSV dengan kolom: [filename, kss_score]
```

**Format labels.csv**:
```
filename,kss_score
1-1.edf,3
1-2.edf,5
...
```

---

## Menjalankan Program

### Training Model
```bash
cd code
python train.py
```

**Konfigurasi training** dapat diubah di [code/config.py](code/config.py):
- `BATCH_SIZE`: Default 32
- `EPOCHS`: Default 50
- `LEARNING_RATE`: Default 1e-4
- `N_SPLITS`: 5-fold cross-validation
- `TASK_TYPE`: "regression" (dapat diubah ke "classification")

### Monitoring Training
Model menggunakan **Weights & Biases (wandb)** untuk tracking. Untuk online logging:
```bash
wandb login
python train.py
```

Untuk offline mode (tanpa login), set di [code/config.py](code/config.py):
```python
os.environ['WANDB_MODE'] = 'offline'
```

---

## How to Cite

Jika repositori ini digunakan pada riset atau publikasi, mohon cantumkan sitasi tugas akhir ini.

### Format IEEE
```text
B. F. Fauzan, "Deteksi Tingkat Kantuk Berdasarkan Sinyal EEG dan EOG Menggunakan Model Mamba: Studi Perbandingan Regresi dan Klasifikasi," Skripsi, Program Studi Teknik Informatika, Institut Teknologi Sumatera, 2026.
```

### Format APA
```text
Fauzan, B. F. (2026). Deteksi Tingkat Kantuk Berdasarkan Sinyal EEG dan EOG Menggunakan Model Mamba Menggunakan Model Mamba: Studi Perbandingan Regresi dan Klasifikasi (Skripsi). Institut Teknologi Sumatera.
```

### BibTeX
```bibtex
@mastersthesis{fauzan2026deteksi,
  author = {Fauzan, Bintang Fikri},
  title = {Deteksi Tingkat Kantuk Berdasarkan Sinyal EEG dan EOG Menggunakan Model Mamba Menggunakan Model Mamba: Studi Perbandingan Regresi dan Klasifikasi},
  school = {Institut Teknologi Sumatera},
  year = {2026},
  type = {Skripsi}
}
```

---

## Referensi

### Paper & Articles
- [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.08956)
- [The ULg multimodality drowsiness database (called DROZY)](https://www.researchgate.net/publication/303563949_The_ULg_multimodality_drowsiness_database_called_DROZY_and_examples_of_use)
- [Optimized driver fatigue detection method using multimodal neural networks](https://www.nature.com/articles/s41598-025-86709-1)

### Library Documentation
- [Mamba Documentation](https://github.com/state-spaces/mamba)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MNE-Python EDF Guide](https://mne.tools/stable/generated/mne.io.read_raw_edf.html)
- [Weights & Biases](https://docs.wandb.ai/)

### Referensi Thesis
Lihat [thesis/references.bib](thesis/references.bib) untuk daftar lengkap referensi yang digunakan.

---

## Troubleshooting

### 1. **CUDA/GPU Error**
```
RuntimeError: CUDA out of memory
```
**Solusi**:
- Kurangi `BATCH_SIZE` di `config.py` (default 32 → 16 atau 8)
- Jalankan pada CPU:
```python
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU
```

### 2. **Module Not Found Error**
```
ModuleNotFoundError: No module named 'mamba_ssm'
```
**Solusi**:
```bash
pip install --upgrade pip setuptools wheel
pip install mamba-ssm==2.2.2 --no-cache-dir
```

### 3. **EDF File Reading Error**
```
FileNotFoundError: Cannot find EDF file
```
**Solusi**:
- Pastikan `Config.DATA_DIR` sudah benar di `config.py`
- Verifikasi file EDF ada di folder `code/psg/`

### 4. **wandb Connection Error**
```
ConnectionError: Failed to connect to wandb
```
**Solusi**:
- Jalankan offline: `wandb offline`
- Atau disable wandb:
```python
os.environ['WANDB_DISABLED'] = 'true'
```

---

## Support & Questions

Jika ada pertanyaan atau menemukan bug:
1. Buka **Issues** di GitHub
2. Sertakan informasi: OS, Python version, error message
3. Lampirkan traceback atau log lengkap

---

**Last Updated**: May 2026
