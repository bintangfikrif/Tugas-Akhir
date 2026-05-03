import torch
import torch.nn as nn
import torchvision.models as models

class MultimodalFeatureCoupledModel(nn.Module):
    def __init__(self, num_classes=3):
        super(MultimodalFeatureCoupledModel, self).__init__()
        
        # ==========================================
        # 1. IMAGE ENCODER: ResNet18
        # ==========================================
        # Inisialisasi ResNet18 bawaan
        self.resnet = models.resnet18(pretrained=False)
        
        # Modifikasi input layer biar nerima 1 channel (Grayscale 75x75) sesuai paper
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modifikasi output layer ResNet biar ngeluarin 512 fitur
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, 512)
        
        # ==========================================
        # 2. TIME-SERIES ENCODER: LSTM
        # ==========================================
        # Input 9 channels, 1 layer, 128 cells
        self.lstm = nn.LSTM(input_size=9, hidden_size=128, num_layers=1, batch_first=True)
        
        # Proyeksi ke 512 fitur agar bisa di-couple dengan output ResNet
        self.lstm_fc = nn.Linear(128, 512)
        
        # ==========================================
        # 3. CLASSIFIER
        # ==========================================
        # Layer klasifikasi akhir untuk 3 kategori (Alert, Low Vigilant, Drowsy)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, img, psg):
        """
        img: (Batch, 1, 75, 75)
        psg: (Batch, 512, 9)
        """
        # --- Ekstraksi Fitur Gambar ---
        img_features = self.resnet(img) # Shape: (Batch, 512)
        
        # --- Ekstraksi Fitur Time-Series (PSG) ---
        lstm_out, (h_n, c_n) = self.lstm(psg)
        # Ambil hidden state terakhir dari sequence
        psg_features = self.lstm_fc(h_n[-1]) # Shape: (Batch, 512)
        
        # --- Min-Max Normalization [0, 1] ---
        # Dilakukan pada masing-masing vektor sebelum digabung
        img_min = img_features.min(dim=1, keepdim=True)[0]
        img_max = img_features.max(dim=1, keepdim=True)[0]
        img_norm = (img_features - img_min) / (img_max - img_min + 1e-8)
        
        psg_min = psg_features.min(dim=1, keepdim=True)[0]
        psg_max = psg_features.max(dim=1, keepdim=True)[0]
        psg_norm = (psg_features - psg_min) / (psg_max - psg_min + 1e-8)
        
        # --- Feature Coupling ---
        # Perkalian elemen menghasilkan vektor coupled dengan panjang 512
        coupled_features = img_norm * psg_norm # Shape: (Batch, 512)
        
        # --- Klasifikasi Akhir ---
        output = self.classifier(coupled_features) # Shape: (Batch, 3)
        
        return output

# --- EVALUASI GFLOPs ---
if __name__ == "__main__":
    from thop import profile, clever_format
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalFeatureCoupledModel().to(device)
    
    # Facial: (Batch=1, Channel=1, H=75, W=75)
    dummy_img = torch.randn(1, 1, 75, 75).to(device)
    # PSG: (Batch=1, Timesteps=512, Channels=9)
    dummy_psg = torch.randn(1, 512, 9).to(device)
    
    # Hitung Kompleksitas
    macs, params = profile(model, inputs=(dummy_img, dummy_psg), verbose=False)
    macs_str, params_str = clever_format([macs, params], "%.3f")
    
    print("\n" + "="*50)
    print("📊 BENCHMARK PAPER REFERENSI (CAO ET AL.)")
    print("="*50)
    print(f"Jumlah Parameter: {params_str} ({(params/1e6):.2f}M)")
    print(f"GFLOPs (MACs):    {macs/1e9:.4f}")
    print("="*50)