import mne
import matplotlib.pyplot as plt

# --- Konfigurasi ---
file_path = 'psg/1-1.edf'
duration_to_show = 20

try:
    raw = mne.io.read_raw_edf(file_path, preload=True)
    channels_to_plot = [
        # 5 Channel EEG
        'Fz', 
        'Cz', 
        'C3',
        'C4',
        'Pz',
        # 2 Channel EOG
        'EOG-V',
        'EOG-H'
    ]
    
    print("\nChannel yang akan ditampilkan:")
    print(channels_to_plot)
    print("-" * 30)

    # Langkah 3: Gunakan daftar nama tersebut di parameter 'picks'
    print("Menampilkan plot interaktif dari MNE untuk channel yang dipilih...")

    raw.plot(picks=channels_to_plot, 
             duration=duration_to_show,
             n_channels=len(channels_to_plot),
             scalings='auto')  
    plt.show()

except ValueError:
    print(f"Error: Salah satu nama channel di 'channels_to_plot' tidak ditemukan di file EDF.")
    print("Harap periksa kembali nama channel Anda dengan daftar yang tersedia.")
except Exception as e:
    print(f"Terjadi error: {e}")