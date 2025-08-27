# ======================================================================================
# GIAI ĐOẠN 3: NÉN VÀ TẠO DATASET CUỐI CÙNG
# ======================================================================================
print("GIAI ĐOẠN 3: NÉN VÀ TẠO DATASET CUỐI CÙNG...")

# --- CÀI ĐẶT VÀ IMPORT ---
!pip install kaggle tqdm -q
import os
import json
from google.colab import drive

# --- CẤU HÌNH ---
class CONFIG:
    # --- Thông tin các dataset trên Kaggle ---
    KAGGLE_USERNAME = "lptoan"
    # Dataset chứa đặc trưng đã xử lý (kết quả từ Script 2)
    PROCESSED_FEATURES_DATASET_ID = "processed-features-dataset"
    # Dataset cuối cùng, hoàn chỉnh sẽ được tạo ra
    FINAL_DATASET_ID = "ngt-ai-torchaudio-features-final"

    # --- Lựa chọn chế độ: False = Tạo mới, True = Cập nhật phiên bản ---
    UPDATE_EXISTING_DATASET = False

    # --- Đường dẫn khác ---
    KAGGLE_JSON_DRIVE_PATH = "/content/drive/MyDrive/Tai_Lieu_NCKH/KaggleAPI/kaggle.json"
    DRIVE_MOUNT_PATH = "/content/drive"

# --- KẾT NỐI VÀ XÁC THỰC ---
drive.mount(CONFIG.DRIVE_MOUNT_PATH, force_remount=True)
!mkdir -p ~/.kaggle
!cp "{CONFIG.KAGGLE_JSON_DRIVE_PATH}" ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
print("Xác thực Kaggle API thành công!")

# --- THIẾT LẬP THƯ MỤC LÀM VIỆC ---
# Thư mục để tải toàn bộ các file đặc trưng từ Kaggle về
DOWNLOADED_FEATURES_PATH = "downloaded_processed_features"
# Thư mục chứa gói tải lên cuối cùng (chỉ có file .7z và metadata.json)
FINAL_UPLOAD_PACKAGE_PATH = "final_upload_package"

os.makedirs(DOWNLOADED_FEATURES_PATH, exist_ok=True)
os.makedirs(FINAL_UPLOAD_PACKAGE_PATH, exist_ok=True)


# --- TẢI TOÀN BỘ DATASET ĐẶC TRƯNG ĐÃ XỬ LÝ ---
print(f"\nBắt đầu tải toàn bộ đặc trưng từ dataset '{CONFIG.PROCESSED_FEATURES_DATASET_ID}'...")
!kaggle datasets download {CONFIG.KAGGLE_USERNAME}/{CONFIG.PROCESSED_FEATURES_DATASET_ID} -p {DOWNLOADED_FEATURES_PATH} --unzip
print("Tải về toàn bộ đặc trưng thành công.")


# --- NÉN THƯ MỤC ĐẶC TRƯNG THÀNH 1 FILE DUY NHẤT ---
output_7z_path = os.path.join(FINAL_UPLOAD_PACKAGE_PATH, f"{CONFIG.FINAL_DATASET_ID}.7z")
!rm -f "{output_7z_path}" # Xóa file nén cũ nếu có

print(f"\nBắt đầu nén thư mục '{DOWNLOADED_FEATURES_PATH}'...")
!sudo apt-get install p7zip-full -qq
!7z a -mx=9 "{output_7z_path}" "{DOWNLOADED_FEATURES_PATH}/"
print(f"Nén thành công! File nén được lưu tại: {output_7z_path}")


# --- TẠO METADATA VÀ TẢI LÊN DATASET CUỐI CÙNG ---
metadata = {
    "title": f"Cough Spectrograms - Torchaudio Features",
    "id": f"{CONFIG.KAGGLE_USERNAME}/{CONFIG.FINAL_DATASET_ID}",
    "licenses": [{"name": "CC0-1.0"}]
}
metadata_path = os.path.join(FINAL_UPLOAD_PACKAGE_PATH, "dataset-metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print("Tạo file metadata.json thành công.")

if CONFIG.UPDATE_EXISTING_DATASET:
    print("\nChế độ: CẬP NHẬT. Bắt đầu tải lên phiên bản mới cho dataset cuối cùng...")
    !kaggle datasets version -p "{FINAL_UPLOAD_PACKAGE_PATH}" -m "Final compressed package of all features"
else:
    print("\nChế độ: TẠO MỚI. Bắt đầu tải lên dataset cuối cùng...")
    !kaggle datasets create -p "{FINAL_UPLOAD_PACKAGE_PATH}"

print(f"\nHOÀN TẤT TOÀN BỘ QUY TRÌNH!")
print(f"Hãy kiểm tra dataset cuối cùng của bạn tại: https://www.kaggle.com/{CONFIG.KAGGLE_USERNAME}/{CONFIG.FINAL_DATASET_ID}")