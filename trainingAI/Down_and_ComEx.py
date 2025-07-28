# -*- coding: utf-8 -*-
"""
Script tự động tải và giải nén dữ liệu đa phần (.001, .002...) từ Google Drive
sử dụng công cụ dòng lệnh 7z để đảm bảo độ tin cậy cao nhất.
"""

# BƯỚC 1: CÀI ĐẶT VÀ IMPORT THƯ VIỆN
print("STEP 1: Cài đặt và import thư viện...")
!pip install -q gdown
!apt-get update -qq && apt-get install -y p7zip-full -qq

from google.colab import drive
import os
import gdown
import glob
import shutil

# BƯỚC 2: KẾT NỐI VỚI GOOGLE DRIVE
print("\nSTEP 2: Kết nối với Google Drive...")
drive.mount('/content/drive')
print("-> Kết nối thành công.")

# --- ⚙️ BƯỚC 3: CẤU HÌNH ---
print("\nSTEP 3: Cấu hình...")
# 1. Dán ID của THƯ MỤC trên Google Drive chứa các file nén
FOLDER_ID = '19OcHdaHqglR75HoKgIUvUFWcKUWwPP3A'

# 2. Tên của FILE NÉN ĐẦU TIÊN (ví dụ: data.7z.001)
MAIN_ARCHIVE_FILENAME = 'ngtai_dataset.zip.111.001'

# 3. Thư mục trên Colab để chứa mọi thứ
DESTINATION_FOLDER = '/content/ngtai_data/'
# ------------------------------------
print("-> Cấu hình hoàn tất.")

# Tạo thư mục làm việc nếu chưa có
os.makedirs(DESTINATION_FOLDER, exist_ok=True)

# --- BƯỚC 4: TẢI DỮ LIỆU ---
print(f"\nSTEP 4: Tải dữ liệu từ Google Drive Folder ID: {FOLDER_ID}...")
gdown.download_folder(id=FOLDER_ID, output=DESTINATION_FOLDER, quiet=False, use_cookies=False)
print(f"-> Đã tải xong các file vào: {DESTINATION_FOLDER}")

# --- BƯỚC 5: GIẢI NÉN BẰNG 7-ZIP ---
print("\nSTEP 5: Bắt đầu quá trình giải nén...")

# Tự động tìm đường dẫn chính xác của file đầu tiên
search_pattern = os.path.join(DESTINATION_FOLDER, '**', MAIN_ARCHIVE_FILENAME)
found_files = glob.glob(search_pattern, recursive=True)

if not found_files:
    raise FileNotFoundError(f"Lỗi: Không tìm thấy file chính '{MAIN_ARCHIVE_FILENAME}' trong '{DESTINATION_FOLDER}'.")

main_archive_filepath = found_files[0]
print(f"-> Tìm thấy file chính tại: {main_archive_filepath}. Bắt đầu giải nén...")

# Chạy lệnh 7z trên file đầu tiên, nó sẽ tự động tìm các phần còn lại
# Giải nén trực tiếp vào thư mục đích (WORKING_DIRECTORY)
!7z x "{main_archive_filepath}" -o"{DESTINATION_FOLDER}" -y
print("-> Giải nén hoàn tất.")

# --- BƯỚC 6: DỌN DẸP FILE NÉN (TÙY CHỌN) ---
print("\nSTEP 6: Dọn dẹp các file nén đã tải về...")
try:
    for f_path in found_files:
        # Tìm và xóa tất cả các file liên quan
        base_name = os.path.splitext(f_path)[0] # ví dụ: /path/to/ngtai_dataset.7z
        for part_file in glob.glob(f"{base_name}.*"):
            os.remove(part_file)
            print(f"-> Đã xóa: {os.path.basename(part_file)}")
except Exception as e:
    print(f"Lỗi khi dọn dẹp file: {e}")


# --- BƯỚC 7: KIỂM TRA KẾT QUẢ CUỐI CÙNG ---
print("\nSTEP 7: Kiểm tra kết quả cuối cùng...")
print(f"Các file và thư mục trong '{DESTINATION_FOLDER}':")
!ls -lha {DESTINATION_FOLDER}