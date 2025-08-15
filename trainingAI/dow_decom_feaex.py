# ======================================================================================
# BLOCK 1: CÀI ĐẶT, IMPORT VÀ CẤU HÌNH (LẤY API TỪ DRIVE)
# ======================================================================================
print("BLOCK 1: CÀI ĐẶT, IMPORT VÀ CẤU HÌNH...")

# Cài đặt thư viện Kaggle và các thư viện khác
!pip install kaggle librosa pydub tqdm matplotlib seaborn pandas -q

from google.colab import drive
import os
import json
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# Cấu hình toàn bộ pipeline
class CONFIG:
    # --- CÁC THÔNG TIN BẠN CUNG CẤP ---
    # 1. Đường dẫn đến file kaggle.json trên Google Drive của bạn
    KAGGLE_JSON_DRIVE_PATH = "/content/drive/MyDrive/Tai_Lieu_NCKH/KaggleAPI/kaggle.json"

    # 2. Lệnh API để tải dataset từ Kaggle
    KAGGLE_API_COMMAND = "kaggle datasets download -d lptoan/ngt-ai-dataset"
    
    # 3. Tên file nén ĐẦU TIÊN (file .zip hoặc .zip.001)
    # <<< ĐÃ SỬA LẠI CHO ĐÚNG >>>
    # Tên file này PHẢI khớp với tên file được tải về từ lệnh API ở trên.
    # Dựa trên lệnh của bạn, tên file tải về sẽ là "ngt-ai-dataset.zip".
    # Nếu dataset của bạn thực sự được chia thành nhiều phần, file đầu tiên có thể là "ngt-ai-dataset.zip.001"
    FIRST_ARCHIVE_FILE_NAME = "ngt-ai-dataset.zip" 
    
    # 4. Thông tin để tải dataset mới lên Kaggle
    KAGGLE_USERNAME = "lptoan" 
    # Tên trên URL (duy nhất, chữ thường, số, dấu gạch ngang)
    DATASET_ID = "processed-ngt-ai-dataset" # Đổi tên để tránh trùng lặp với dataset gốc
    # ---------------------------------------------

    # Các cấu hình khác
    DRIVE_MOUNT_PATH = "/content/drive"

# --- Kết nối Google Drive và Xác thực Kaggle API ---
drive.mount(CONFIG.DRIVE_MOUNT_PATH)

print("\nBắt đầu xác thực Kaggle API từ Google Drive...")
if not os.path.exists(CONFIG.KAGGLE_JSON_DRIVE_PATH):
    print(f"LỖI: Không tìm thấy file 'kaggle.json' tại đường dẫn: {CONFIG.KAGGLE_JSON_DRIVE_PATH}")
    print("Vui lòng kiểm tra lại đường dẫn hoặc đảm bảo bạn đã tải file lên Google Drive.")
else:
    !mkdir -p ~/.kaggle
    !cp "{CONFIG.KAGGLE_JSON_DRIVE_PATH}" ~/.kaggle/kaggle.json
    !chmod 600 ~/.kaggle/kaggle.json
    print("Xác thực Kaggle API thành công!")


# ======================================================================================
# BLOCK 2: TẢI VÀ GIẢI NÉN DATASET BẰNG 7-ZIP
# ======================================================================================
print("\nBLOCK 2: TẢI VÀ GIẢI NÉN DATASET...")

DATA_FOLDER = "kaggle_data"
os.makedirs(DATA_FOLDER, exist_ok=True)

print(f"Bắt đầu tải dataset...")
!{CONFIG.KAGGLE_API_COMMAND}

print("\nĐang cài đặt 7-Zip...")
!sudo apt-get install p7zip-full -qq

# 1. Giải nén file lớn (file gốc)
print(f"Bắt đầu giải nén file lớn '{CONFIG.FIRST_ARCHIVE_FILE_NAME}'...")
!7z x '{CONFIG.FIRST_ARCHIVE_FILE_NAME}' -o'{DATA_FOLDER}'
print("Đã giải nén file lớn thành công. Đang dọn dẹp...")
!rm *.zip* # Xóa các file nén gốc

# 2. Tìm và giải nén các file zip con (nếu có)
print("\nBắt đầu tìm và giải nén các file zip con...")
zip_files_to_extract = []
for root, dirs, files in os.walk(DATA_FOLDER):
    for file in files:
        if file.endswith(".zip"):
            zip_files_to_extract.append(os.path.join(root, file))

if not zip_files_to_extract:
    print("Không tìm thấy file zip con nào để giải nén thêm.")
else:
    print(f"Tìm thấy {len(zip_files_to_extract)} file zip con. Bắt đầu giải nén...")
    for zip_file in zip_files_to_extract:
        extract_to_folder = os.path.splitext(zip_file)[0]
        os.makedirs(extract_to_folder, exist_ok=True)
        print(f"  -> Đang giải nén '{zip_file}' vào '{extract_to_folder}'")
        !unzip -q '{zip_file}' -d '{extract_to_folder}'
        !rm '{zip_file}'
    print("Đã giải nén và dọn dẹp tất cả các file zip con.")

print("\nHoàn tất toàn bộ quá trình tải và giải nén!")


# ======================================================================================
# BLOCK 3: TRÍCH XUẤT ĐẶC TRƯNG LOG-MEL SPECTROGRAM
# ======================================================================================
print("\nBLOCK 3: TRÍCH XUẤT ĐẶC TRƯNG LOG-MEL SPECTROGRAM...")

class FeatureConfig:
    SAMPLE_RATE = 16000
    N_MELS = 128
    HOP_LENGTH = 512
    N_FFT = 2048

FEATURE_FOLDER = "feature_spectrograms"
os.makedirs(FEATURE_FOLDER, exist_ok=True)
print(f"Các đặc trưng sẽ được lưu tại thư mục: '{FEATURE_FOLDER}'")

audio_files = []
for root, dirs, files in os.walk(DATA_FOLDER):
    for file in files:
        if file.endswith((".wav", ".flac", ".mp3")):
            audio_files.append(os.path.join(root, file))
print(f"Tìm thấy {len(audio_files)} file âm thanh để xử lý.")

for audio_path in tqdm(audio_files, desc="Trích xuất đặc trưng"):
    try:
        y, sr = librosa.load(audio_path, sr=FeatureConfig.SAMPLE_RATE, mono=True)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=FeatureConfig.N_FFT, hop_length=FeatureConfig.HOP_LENGTH, n_mels=FeatureConfig.N_MELS)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        relative_path = os.path.relpath(audio_path, DATA_FOLDER)
        base_name = os.path.splitext(relative_path)[0].replace(os.sep, '_')
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_mel_spectrogram, sr=sr, hop_length=FeatureConfig.HOP_LENGTH, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(f'Log-Mel Spectrogram: {os.path.basename(audio_path)}')
        plt.tight_layout()
        image_path = os.path.join(FEATURE_FOLDER, f"{base_name}.png")
        plt.savefig(image_path)
        plt.close()
        
        numpy_path = os.path.join(FEATURE_FOLDER, f"{base_name}.npy")
        np.save(numpy_path, log_mel_spectrogram)
    except Exception as e:
        print(f"Lỗi khi xử lý file {audio_path}: {e}")

print("\nHoàn tất quá trình trích xuất đặc trưng!")

# ======================================================================================
# BLOCK 4: NÉN "ULTRA" VÀ TẢI ĐẶC TRƯNG LÊN KAGGLE
# ======================================================================================
print("\nBLOCK 4: NÉN 'ULTRA' VÀ TẢI ĐẶC TRƯNG LÊN KAGGLE...")

KAGGLE_UPLOAD_FOLDER = "kaggle_upload"
os.makedirs(KAGGLE_UPLOAD_FOLDER, exist_ok=True)

print(f"Bắt đầu nén thư mục '{FEATURE_FOLDER}' ở chế độ Ultra...")
output_7z_path = os.path.join(KAGGLE_UPLOAD_FOLDER, f"{CONFIG.DATASET_ID}.7z")
!7z a -mx=9 {output_7z_path} ./{FEATURE_FOLDER}/*
print(f"Đã nén thành công, file lưu tại: {output_7z_path}")

print("Đang tạo file 'dataset-metadata.json'...")
metadata = {
  "title": f"Processed Cough Spectrograms ({CONFIG.DATASET_ID})",
  "id": f"{CONFIG.KAGGLE_USERNAME}/{CONFIG.DATASET_ID}",
  "licenses": [{"name": "CC0-1.0"}]
}
metadata_path = os.path.join(KAGGLE_UPLOAD_FOLDER, "dataset-metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print("Tạo file metadata thành công.")

print("Bắt đầu tải dataset lên Kaggle...")
!kaggle datasets create -p {KAGGLE_UPLOAD_FOLDER} --dir-mode zip
print("\nHoàn tất việc tải dataset lên Kaggle!")
print(f"Hãy kiểm tra trang Kaggle của bạn để xem dataset mới tại: https://www.kaggle.com/{CONFIG.KAGGLE_USERNAME}/{CONFIG.DATASET_ID}")