# ======================================================================================
# PHIÊN BẢN SCRIPT TỐI ƯU - SỬ DỤNG TORCHAUDIO & LỌC NHIỄU
# ======================================================================================

# ======================================================================================
# BLOCK 1: CÀI ĐẶT, IMPORT VÀ CẤU HÌNH
# ======================================================================================
print("BLOCK 1: CÀI ĐẶT, IMPORT VÀ CẤU HÌNH...")

# Cài đặt các thư viện cần thiết
!pip install kaggle librosa tqdm pandas torch torchaudio noisereduce -q

from google.colab import drive
import os
import json
import librosa
import numpy as np
from tqdm.auto import tqdm
import sys
import torch
import torchaudio # <<< THAY ĐỔI: Sử dụng torchaudio
import torchaudio.transforms as T # <<< THAY ĐỔI: Import transforms
import noisereduce as nr

# --- CẤU HÌNH TOÀN BỘ PIPELINE ---
class CONFIG:
    # --- CÁC THÔNG TIN BẠN CẦN CUNG CẤP ---
    UPDATE_EXISTING_DATASET = False
    KAGGLE_JSON_DRIVE_PATH = "/content/drive/MyDrive/Tai_Lieu_NCKH/KaggleAPI/kaggle.json"
    KAGGLE_API_COMMAND = "kaggle datasets download -d lptoan/ngt-ai-dataset"
    FIRST_ARCHIVE_FILE_NAME = "ngt-ai-dataset.zip"
    KAGGLE_USERNAME = "lptoan"
    DATASET_ID = "ngt-ai-torchaudio-features" # <<< THAY ĐỔI >>> Đặt tên mới
    VALID_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a")
    # ---------------------------------------------------
    DRIVE_MOUNT_PATH = "/content/drive"

# --- Kết nối Google Drive và Xác thực Kaggle API ---
drive.mount(CONFIG.DRIVE_MOUNT_PATH, force_remount=True)
print("\nBắt đầu xác thực Kaggle API từ Google Drive...")
if not os.path.exists(CONFIG.KAGGLE_JSON_DRIVE_PATH):
    print(f"LỖI: Không tìm thấy file 'kaggle.json' tại: {CONFIG.KAGGLE_JSON_DRIVE_PATH}")
    sys.exit("Dừng script vì thiếu file API.")
else:
    !mkdir -p ~/.kaggle
    !cp "{CONFIG.KAGGLE_JSON_DRIVE_PATH}" ~/.kaggle/kaggle.json
    !chmod 600 ~/.kaggle/kaggle.json
    print("Xác thực Kaggle API thành công!")

# ======================================================================================
# BLOCK 2: TẢI VÀ GIẢI NÉN DATASET
# ======================================================================================
print("\nBLOCK 2: TẢI VÀ GIẢI NÉN DATASET...")
DATA_FOLDER = "kaggle_data"
os.makedirs(DATA_FOLDER, exist_ok=True)
!{CONFIG.KAGGLE_API_COMMAND}
!sudo apt-get install p7zip-full -qq
print(f"\nBắt đầu giải nén file lớn '{CONFIG.FIRST_ARCHIVE_FILE_NAME}'...")
!7z x -y "{CONFIG.FIRST_ARCHIVE_FILE_NAME}" -o"{DATA_FOLDER}"
print("Đã giải nén file lớn. Đang dọn dẹp file zip gốc...")
!rm -f "{CONFIG.FIRST_ARCHIVE_FILE_NAME}"
print("\nBắt đầu tìm và giải nén file split (nếu có)...")
first_split_file = next((os.path.join(DATA_FOLDER, f) for f in os.listdir(DATA_FOLDER) if f.endswith(".zip.001")), None)
if first_split_file:
    print(f"Tìm thấy file split: '{first_split_file}'. Bắt đầu giải nén...")
    !7z x -y "{first_split_file}" -o"{DATA_FOLDER}"
    print("Đã giải nén file split thành công. Đang dọn dẹp các mảnh nén...")
    !rm -f {DATA_FOLDER}/*.zip.*
else:
    print("Không tìm thấy file split lồng nhau nào, bỏ qua bước này.")
print("\nHoàn tất toàn bộ quá trình tải và giải nén!")


# ======================================================================================
# BLOCK 3: TRÍCH XUẤT ĐẶC TRƯNG VỚI LỌC NHIỄU VÀ TORCHAUDIO
# ======================================================================================
print("\nBLOCK 3: TRÍCH XUẤT ĐẶC TRƯNG...")

# --- Cấu hình cho Mel Spectrogram ---
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

FEATURE_FOLDER = "feature_spectrograms"
os.makedirs(FEATURE_FOLDER, exist_ok=True)

# <<< THAY ĐỔI >>> Khởi tạo lớp MelSpectrogram của torchaudio
# Lớp này hoạt động như một tầng mạng neural, có thể chạy trên GPU
mel_spectrogram_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LENGTH,
    n_mels=N_MELS
)
print("Đã khởi tạo lớp trích xuất đặc trưng của torchaudio.")

# --- Tìm tất cả các file âm thanh ---
audio_files = []
for root, dirs, files in os.walk(DATA_FOLDER):
    for file in files:
        if file.endswith(CONFIG.VALID_AUDIO_EXTENSIONS):
            audio_files.append(os.path.join(root, file))

if not audio_files:
    print(f"CẢNH BÁO: Không tìm thấy file âm thanh nào trong thư mục '{DATA_FOLDER}'.")
    sys.exit("Dừng script vì không có dữ liệu để xử lý.")
else:
    print(f"Tìm thấy {len(audio_files)} file âm thanh. Bắt đầu xử lý...")
    for audio_path in tqdm(audio_files, desc="Trích xuất đặc trưng"):
        try:
            # 1. Tải file âm thanh
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            
            # 2. Lọc nhiễu
            y_clean = nr.reduce_noise(y=y, sr=sr)
            
            # 3. Trích xuất đặc trưng bằng torchaudio
            audio_tensor = torch.tensor(y_clean, dtype=torch.float32)
            
            mel_spectrogram = mel_spectrogram_transform(audio_tensor)
            
            # Chuyển sang thang đo dB (Log-Mel)
            log_mel_spectrogram = T.AmplitudeToDB()(mel_spectrogram)
            
            # 4. Lưu kết quả
            spectrogram_numpy = log_mel_spectrogram.squeeze().numpy()
            
            relative_path = os.path.relpath(audio_path, DATA_FOLDER)
            base_name = os.path.splitext(relative_path)[0].replace(os.sep, '_')
            numpy_path = os.path.join(FEATURE_FOLDER, f"{base_name}.npy")
            np.save(numpy_path, spectrogram_numpy)

        except Exception as e:
            print(f"Lỗi khi xử lý file {audio_path}: {e}")
    print("\nHoàn tất quá trình trích xuất đặc trưng!")

# ======================================================================================
# BLOCK 4: NÉN VÀ TẢI ĐẶC TRƯNG LÊN KAGGLE
# ======================================================================================
print("\nBLOCK 4: NÉN 'ULTRA' VÀ TẢI ĐẶC TRƯNG LÊN KAGGLE...")
KAGGLE_UPLOAD_FOLDER = "kaggle_upload"
os.makedirs(KAGGLE_UPLOAD_FOLDER, exist_ok=True)
output_7z_path = os.path.join(KAGGLE_UPLOAD_FOLDER, f"{CONFIG.DATASET_ID}.7z")
!rm -f {output_7z_path}
print(f"Bắt đầu nén thư mục '{FEATURE_FOLDER}' ở chế độ Ultra...")
!7z a -mx=9 {output_7z_path} ./{FEATURE_FOLDER}/
print(f"Đã nén thành công, file lưu tại: {output_7z_path}")
metadata = {
    "title": f"Torchaudio Features ({CONFIG.DATASET_ID})",
    "id": f"{CONFIG.KAGGLE_USERNAME}/{CONFIG.DATASET_ID}",
    "licenses": [{"name": "CC0-1.0"}]
}
metadata_path = os.path.join(KAGGLE_UPLOAD_FOLDER, "dataset-metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print("Tạo file metadata.json thành công.")
if CONFIG.UPDATE_EXISTING_DATASET:
    print("\nChế độ: CẬP NHẬT. Bắt đầu tải lên phiên bản mới...")
    !kaggle datasets version -p "{KAGGLE_UPLOAD_FOLDER}" -m "Updated with torchaudio features" --dir-mode zip
    print(f"\nHoàn tất! Kiểm tra phiên bản mới tại: https://www.kaggle.com/{CONFIG.KAGGLE_USERNAME}/{CONFIG.DATASET_ID}")
else:
    print("\nChế độ: TẠO MỚI. Bắt đầu tải dataset mới lên...")
    !kaggle datasets create -p "{KAGGLE_UPLOAD_FOLDER}" --dir-mode zip
    print(f"\nHoàn tất! Kiểm tra dataset mới tại: https://www.kaggle.com/{CONFIG.KAGGLE_USERNAME}/{CONFIG.DATASET_ID}")