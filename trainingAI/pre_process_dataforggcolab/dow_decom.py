# ======================================================================================
# GIAI ĐOẠN 1: TẢI, GIẢI NÉN, NÉN ULTRA VÀ TẠO DATASET THÔ
# ======================================================================================
print("GIAI ĐOẠN 1: TẢI, GIẢI NÉN, NÉN ULTRA VÀ TẠO DATASET THÔ...")

import os
import json
from google.colab import drive

# --- CẤU HÌNH ---
class CONFIG:
    KAGGLE_API_COMMAND_ORIGINAL = "kaggle datasets download -d lptoan/ngt-ai-dataset"
    FIRST_ARCHIVE_FILE_NAME = "ngt-ai-dataset.zip"
    KAGGLE_USERNAME = "lptoan"
    RAW_DATASET_ID = "raw-audio-dataset" 
    KAGGLE_JSON_DRIVE_PATH = "/content/drive/MyDrive/Tai_Lieu_NCKH/KaggleAPI/kaggle.json"
    DRIVE_MOUNT_PATH = "/content/drive"

# --- KẾT NỐI VÀ XÁC THỰC ---
drive.mount(CONFIG.DRIVE_MOUNT_PATH, force_remount=True)
!mkdir -p ~/.kaggle
!cp "{CONFIG.KAGGLE_JSON_DRIVE_PATH}" ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
print("Xác thực Kaggle API thành công!")

# --- TẢI VÀ GIẢI NÉN VÀO THƯ MỤC TẠM ---
RAW_AUDIO_LOCAL_FOLDER = "raw_audio_local"
os.makedirs(RAW_AUDIO_LOCAL_FOLDER, exist_ok=True)

print("\nBắt đầu tải dataset gốc...")
!{CONFIG.KAGGLE_API_COMMAND_ORIGINAL}

print("\nBắt đầu giải nén...")
!sudo apt-get install p7zip-full -qq
!7z x -y "{CONFIG.FIRST_ARCHIVE_FILE_NAME}" -o"{RAW_AUDIO_LOCAL_FOLDER}"
!rm -f "{CONFIG.FIRST_ARCHIVE_FILE_NAME}"
first_split_file = next((os.path.join(RAW_AUDIO_LOCAL_FOLDER, f) for f in os.listdir(RAW_AUDIO_LOCAL_FOLDER) if f.endswith(".zip.001")), None)
if first_split_file:
    !7z x -y "{first_split_file}" -o"{RAW_AUDIO_LOCAL_FOLDER}"
    !rm -f {RAW_AUDIO_LOCAL_FOLDER}/*.zip.*
print("Giải nén hoàn tất. Toàn bộ file âm thanh thô đã sẵn sàng.")

# --- TỰ NÉN TOÀN BỘ DỮ LIỆU THÔ THÀNH 1 FILE .7z ---
UPLOAD_PACKAGE_FOLDER = "upload_package_raw"
os.makedirs(UPLOAD_PACKAGE_FOLDER, exist_ok=True)
output_7z_path = os.path.join(UPLOAD_PACKAGE_FOLDER, "raw_audio.7z")

print(f"\nBắt đầu nén toàn bộ thư mục '{RAW_AUDIO_LOCAL_FOLDER}' ở chế độ ULTRA...")
# <<< THAY ĐỔI Ở ĐÂY: Sử dụng -mx=9 cho chế độ nén Ultra >>>
!7z a -mx=9 "{output_7z_path}" "{RAW_AUDIO_LOCAL_FOLDER}/"
print(f"Nén hoàn tất! File lưu tại: {output_7z_path}")

# --- TẠO METADATA VÀ TẢI GÓI NÉN LÊN KAGGLE ---
metadata = {
    "title": f"Raw Audio Dataset ({CONFIG.RAW_DATASET_ID})",
    "id": f"{CONFIG.KAGGLE_USERNAME}/{CONFIG.RAW_DATASET_ID}",
    "licenses": [{"name": "CC0-1.0"}],
    "private": False    
}
metadata_path = os.path.join(UPLOAD_PACKAGE_FOLDER, "dataset-metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"\nBắt đầu tải gói nén '{output_7z_path}' lên Kaggle...")
!kaggle datasets create -p "{UPLOAD_PACKAGE_FOLDER}"

print(f"\nHOÀN TẤT GIAI ĐOẠN 1!")
print(f"Hãy kiểm tra dataset riêng tư mới của bạn tại: https://www.kaggle.com/{CONFIG.KAGGLE_USERNAME}/{CONFIG.RAW_DATASET_ID}")