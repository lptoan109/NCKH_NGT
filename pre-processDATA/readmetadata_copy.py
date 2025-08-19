import pandas as pd
import os
import shutil
from pathlib import Path

# ==============================================================================
# --- PHẦN CẤU HÌNH ---
# Thay đổi các giá trị dưới đây để điều khiển script.
# ==============================================================================

# 1. CÁC CÔNG TẮC CHỨC NĂNG (True: Bật, False: Tắt)

# --- Chế độ tìm kiếm file ---
# True: Dành cho bộ dữ liệu như 'TB Screen Dataset'. Script sẽ tìm TẤT CẢ các file có tên BẮT ĐẦU bằng ID trong cột ID_COLUMN.
# False: Dành cho bộ dữ liệu như 'UK Covid-19'. Script sẽ đọc tên file đầy đủ từ các cột trong AUDIO_COLUMNS.
FIND_FILES_BY_ID_PREFIX = False

PROCESS_MULTIPLE_SOURCES = True  # True nếu bạn có nhiều thư mục nguồn. False nếu chỉ có một.
RENAME_FILES = False             # True để đổi tên file thành 'ID_ten_file_goc.wav'.
                                 # (Công tắc này không có tác dụng khi FIND_FILES_BY_ID_PREFIX = True)

# 2. CẤU HÌNH ĐƯỜNG DẪN
EXCEL_FILE_PATH = Path(r'H:\Toàn-Khang_NCKH\ngtai_dataset\newmetadata_completed.xlsx')
DESTINATION_FOLDER = Path(r'H:\Toàn-Khang_NCKH\ngtai_dataset\Lỗi')

# --- Cấu hình thư mục nguồn ---
SOURCE_AUDIO_FOLDERS = [
    Path(r'H:\Toàn-Khang_NCKH\ngtai_dataset\dataset_unhealthy'),
    Path(r'F:\NCKH_Toàn_Khang\TBSCREENDATASET\TBscreen_Dataset\Passive_coughs\Audio_files'),
]

# 3. CẤU HÌNH EXCEL
SHEETS_TO_PROCESS = ['eror']

# --- Tên các cột trong file Excel ---
ID_COLUMN = 'Lớp'
# (Danh sách này chỉ được sử dụng khi FIND_FILES_BY_ID_PREFIX = False)
AUDIO_COLUMNS = ['Tên file']

# ==============================================================================
# --- LOGIC CHÍNH CỦA SCRIPT ---
# (Bạn không cần sửa đổi phần dưới này)
# ==============================================================================

def find_source_file_recursively(filename, source_folders):
    """
    Tìm kiếm một file có tên chính xác trong danh sách các thư mục nguồn,
    bao gồm cả các thư mục con.
    """
    for folder in source_folders:
        # Sử dụng rglob để tìm kiếm đệ quy
        found_files = list(folder.rglob(filename))
        if found_files:
            return found_files[0] # Trả về đường dẫn đầu tiên tìm thấy
    return None

def process_files(config):
    """Đọc file Excel, tạo thư mục con và sao chép file dựa trên cấu hình."""
    source_folders_to_search = config['SOURCE_AUDIO_FOLDERS']
    if not config['PROCESS_MULTIPLE_SOURCES']:
        source_folders_to_search = [config['SOURCE_AUDIO_FOLDERS'][0]]
    
    print("--- Bắt đầu quá trình xử lý ---")
    print(f"File Excel nguồn: {config['EXCEL_FILE_PATH']}")
    print(f"Thư mục đích: {config['DESTINATION_FOLDER']}")
    if config['FIND_FILES_BY_ID_PREFIX']:
        print(">> CHẾ ĐỘ: Tìm file theo tiền tố ID (TB Screen Dataset).")
    else:
        print(">> CHẾ ĐỘ: Tìm file theo tên đầy đủ (UK Covid-19).")
        print(f"   Đổi tên file: {'Bật' if config['RENAME_FILES'] else 'Tắt'}")
    
    config['DESTINATION_FOLDER'].mkdir(parents=True, exist_ok=True)

    total_files_copied = 0
    total_files_not_found = 0

    for sheet_name in config['SHEETS_TO_PROCESS']:
        try:
            df = pd.read_excel(config['EXCEL_FILE_PATH'], sheet_name=sheet_name)
            print(f"\n--- Đang xử lý sheet: '{sheet_name}' ({len(df)} hàng) ---")
        except Exception as e:
            print(f"\nCảnh báo: Không thể đọc sheet '{sheet_name}'. Lỗi: {e}. Bỏ qua.")
            continue
            
        sheet_specific_folder = config['DESTINATION_FOLDER'] / sheet_name
        sheet_specific_folder.mkdir(exist_ok=True)

        for index, row in df.iterrows():
            participant_id = row.get(config['ID_COLUMN'])
            # Bỏ qua nếu ID là rỗng hoặc không phải là chuỗi
            if pd.isna(participant_id):
                continue
            participant_id = str(participant_id)

            # --- LOGIC ĐIỀU KHIỂN BỞI CÔNG TẮC ---
            if config['FIND_FILES_BY_ID_PREFIX']:
                # CHẾ ĐỘ 1: Tìm tất cả file bắt đầu bằng ID (TÌM KIẾM ĐỆ QUY)
                files_found_for_id = 0
                for folder in source_folders_to_search:
                    if not folder.is_dir(): continue
                    # *** THAY ĐỔI CHÍNH: Sử dụng rglob để tìm kiếm trong tất cả thư mục con ***
                    for source_file_path in folder.rglob(f'{participant_id}*'):
                        if source_file_path.is_file():
                            destination_file_path = sheet_specific_folder / source_file_path.name
                            try:
                                shutil.copy2(source_file_path, destination_file_path)
                                total_files_copied += 1
                                files_found_for_id += 1
                            except Exception as e:
                                print(f"  [LỖI SAO CHÉP] {source_file_path.name}: {e}")
                if files_found_for_id == 0:
                    print(f"  [LỖI TÌM KIẾM] Không tìm thấy file nào cho ID '{participant_id}'")
                    total_files_not_found += 1

            else:
                # CHẾ ĐỘ 2: Tìm file theo tên đầy đủ trong các cột (TÌM KIẾM ĐỆ QUY)
                for audio_col in config['AUDIO_COLUMNS']:
                    original_filename = row.get(audio_col)
                    if pd.isna(original_filename) or not isinstance(original_filename, str):
                        continue

                    # *** THAY ĐỔI CHÍNH: Sử dụng hàm tìm kiếm đệ quy mới ***
                    source_file_path = find_source_file_recursively(original_filename, source_folders_to_search)

                    if source_file_path:
                        if config['RENAME_FILES']:
                            new_filename = f"{participant_id}_{original_filename}"
                        else:
                            new_filename = original_filename
                        destination_file_path = sheet_specific_folder / new_filename
                        try:
                            shutil.copy2(source_file_path, destination_file_path)
                            total_files_copied += 1
                        except Exception as e:
                            print(f"  [LỖI SAO CHÉP] {original_filename}: {e}")
                    else:
                        print(f"  [LỖI TÌM KIẾM] Không tìm thấy file '{original_filename}'")
                        total_files_not_found += 1
    
    print(f"\n--- HOÀN TẤT ---")
    print(f"Tổng số file đã sao chép: {total_files_copied}")
    print(f"Tổng số file không tìm thấy (hoặc ID không có file): {total_files_not_found}")
    print(f"Các file đã xử lý được lưu trong '{config['DESTINATION_FOLDER']}'")

if __name__ == "__main__":
    config_settings = {
        'PROCESS_MULTIPLE_SOURCES': PROCESS_MULTIPLE_SOURCES,
        'RENAME_FILES': RENAME_FILES,
        'FIND_FILES_BY_ID_PREFIX': FIND_FILES_BY_ID_PREFIX,
        'EXCEL_FILE_PATH': EXCEL_FILE_PATH,
        'DESTINATION_FOLDER': DESTINATION_FOLDER,
        'SOURCE_AUDIO_FOLDERS': SOURCE_AUDIO_FOLDERS,
        'SHEETS_TO_PROCESS': SHEETS_TO_PROCESS,
        'ID_COLUMN': ID_COLUMN,
        'AUDIO_COLUMNS': AUDIO_COLUMNS
    }
    process_files(config_settings)