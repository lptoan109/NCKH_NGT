import pandas as pd
import shutil
from pathlib import Path

# ==============================================================================
# --- PHẦN CẤU HÌNH ---
# Thay đổi các giá trị dưới đây để điều khiển script.
# ==============================================================================

# 1. ĐƯỜNG DẪN
# Đường dẫn đầy đủ đến file Excel của bạn
EXCEL_FILE_PATH = Path(r'H:\Toàn-Khang_NCKH\ngtai_dataset\newmetadata_completed.xlsx')

# Thư mục gốc chứa các file bạn muốn tìm kiếm (script sẽ tìm cả trong các thư mục con)
SOURCE_FOLDER = Path(r'H:\Toàn-Khang_NCKH\ngtai_dataset\dataset_unhealthy')

# Thư mục nơi các file tìm thấy sẽ được di chuyển đến
DESTINATION_FOLDER = Path(r'H:\Toàn-Khang_NCKH\ngtai_dataset\Lỗi')

# 2. CẤU HÌNH EXCEL
# Danh sách các tên sheet cần được xử lý trong file Excel
# Ví dụ: ['Sheet1', 'Sheet2', 'DuLieuLoi']
SHEETS_TO_PROCESS = ['eror']

# Danh sách tên các cột trong Excel chứa tên file cần tìm kiếm
# Ví dụ: ['TenFileAmThanh', 'FileGoc', 'Path']
FILENAME_COLUMNS = ['Tên file']


# ==============================================================================
# --- LOGIC CHÍNH CỦA SCRIPT ---
# (Bạn không cần sửa đổi phần dưới này)
# ==============================================================================

def find_file_recursively(filename, source_root):
    """
    Tìm kiếm đệ quy một file trong thư mục nguồn.
    Trả về đường dẫn Path đầu tiên tìm thấy hoặc None nếu không có.
    """
    # Sử dụng rglob để tìm kiếm trong tất cả thư mục con
    # Dùng list() để thực hiện tìm kiếm ngay lập tức
    found_files = list(source_root.rglob(filename))
    if found_files:
        return found_files[0]  # Trả về kết quả đầu tiên
    return None

def process_and_move_files():
    """
    Đọc file Excel, tìm kiếm và di chuyển các file được chỉ định.
    """
    # Kiểm tra các đường dẫn cấu hình
    if not EXCEL_FILE_PATH.is_file():
        print(f"[LỖI] Không tìm thấy file Excel tại: {EXCEL_FILE_PATH}")
        return
    if not SOURCE_FOLDER.is_dir():
        print(f"[LỖI] Không tìm thấy thư mục nguồn tại: {SOURCE_FOLDER}")
        return

    # Tạo thư mục đích nếu nó chưa tồn tại
    DESTINATION_FOLDER.mkdir(parents=True, exist_ok=True)

    print("--- BẮT ĐẦU QUÁ TRÌNH TÌM KIẾM VÀ DI CHUYỂN FILE ---")
    print(f"File Excel: {EXCEL_FILE_PATH}")
    print(f"Thư mục nguồn (tìm kiếm sâu): {SOURCE_FOLDER}")
    print(f"Thư mục đích: {DESTINATION_FOLDER}")
    
    # Bộ đếm thống kê
    files_moved_count = 0
    files_not_found_count = 0
    total_files_to_find = 0

    # Lặp qua từng sheet được chỉ định
    for sheet_name in SHEETS_TO_PROCESS:
        try:
            df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=sheet_name, dtype=str).fillna('')
            print(f"\n--- Đang xử lý sheet: '{sheet_name}' ({len(df)} hàng) ---")
        except Exception as e:
            print(f"\n[CẢNH BÁO] Không thể đọc sheet '{sheet_name}'. Lỗi: {e}. Bỏ qua sheet này.")
            continue

        # Lặp qua từng hàng trong DataFrame của sheet hiện tại
        for index, row in df.iterrows():
            # Lặp qua từng cột chứa tên file được chỉ định
            for column_name in FILENAME_COLUMNS:
                if column_name not in df.columns:
                    print(f"[CẢNH BÁO] Cột '{column_name}' không tồn tại trong sheet '{sheet_name}'. Bỏ qua cột này.")
                    break # Chuyển sang sheet tiếp theo nếu cột không tồn tại

                filename = row[column_name]

                # Bỏ qua nếu ô trống hoặc không có giá trị
                if not filename or pd.isna(filename):
                    continue
                
                total_files_to_find += 1
                
                print(f"Đang tìm kiếm file: '{filename}'...")

                # Thực hiện tìm kiếm sâu
                source_path = find_file_recursively(filename, SOURCE_FOLDER)

                if source_path:
                    destination_path = DESTINATION_FOLDER / source_path.name
                    try:
                        # Thực hiện di chuyển file
                        shutil.move(str(source_path), str(destination_path))
                        print(f"  -> [THÀNH CÔNG] Đã di chuyển '{source_path.name}' đến '{DESTINATION_FOLDER}'")
                        files_moved_count += 1
                    except Exception as e:
                        print(f"  -> [LỖI DI CHUYỂN] Không thể di chuyển file '{filename}'. Lỗi: {e}")
                else:
                    # Thông báo nếu không tìm thấy file
                    print(f"  -> [KHÔNG TÌM THẤY] File '{filename}' trong thư mục nguồn.")
                    files_not_found_count += 1

    print("\n--- HOÀN TẤT ---")
    print(f"Tổng số file cần tìm từ Excel: {total_files_to_find}")
    print(f"Tổng số file đã di chuyển thành công: {files_moved_count}")
    print(f"Tổng số file không tìm thấy: {files_not_found_count}")

if __name__ == "__main__":
    process_and_move_files()