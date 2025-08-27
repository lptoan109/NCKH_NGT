import os
import pandas as pd
from pydub import AudioSegment
from pydub.utils import mediainfo
import numpy as np # MỚI: Thêm thư viện numpy để xử lý -inf

# --- CẤU HÌNH ---
# MỚI: Ngưỡng âm thanh để xác định file im lặng (tính bằng dBFS)
# Giá trị càng gần 0 thì âm thanh càng lớn. -90 dBFS là gần như im lặng tuyệt đối.
# Bạn có thể điều chỉnh giá trị này, ví dụ: -50, -60, -70
SILENCE_THRESHOLD = -60.0

def analyze_audio_directory(root_folder):
    """
    Quét qua một thư mục gốc chứa các thư mục con (lớp) và thống kê
    các file âm thanh bên trong.

    Args:
        root_folder (str): Đường dẫn đến thư mục gốc cần quét.

    Returns:
        pandas.DataFrame: Một DataFrame chứa thông tin thống kê.
    """
    audio_data = []
    supported_formats = ['.wav', '.mp3', '.flac'] # Có thể thêm các định dạng khác

    print(f"Bắt đầu quét thư mục: {root_folder}\n...")

    for dirpath, _, filenames in os.walk(root_folder):
        class_name = os.path.basename(dirpath)

        for filename in filenames:
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext in supported_formats:
                file_path = os.path.join(dirpath, filename)
                duration_seconds = 'Lỗi xử lý'
                completeness = 'Hỏng' # MỚI: Mặc định là 'Hỏng'

                try:
                    # Lấy thông tin thời lượng nhanh chóng
                    info = mediainfo(file_path)
                    duration_seconds = round(float(info['duration']), 2)

                    # MỚI: Kiểm tra mức độ hoàn chỉnh bằng cách phân tích âm lượng
                    # Phải load file để kiểm tra âm lượng
                    audio = AudioSegment.from_file(file_path)
                    
                    # audio.dBFS trả về mức năng lượng trung bình
                    # Nếu file hoàn toàn im lặng, nó sẽ trả về -inf
                    loudness = audio.dBFS
                    
                    # Xử lý trường hợp -inf
                    if loudness == -np.inf or loudness < SILENCE_THRESHOLD:
                        completeness = 'Gần như im lặng'
                    else:
                        completeness = 'Hoàn chỉnh'
                    
                    print(f"Đã xử lý: {file_path} ({completeness})")

                except Exception as e:
                    print(f"LỖI: Không thể xử lý file {file_path}. Lý do: {e}")
                    # Trạng thái đã được mặc định là 'Hỏng'
                    pass
                
                # Thu thập dữ liệu
                audio_data.append({
                    'Lớp': class_name,
                    'Tên file': filename,
                    'Định dạng': file_ext,
                    'Thời gian (giây)': duration_seconds,
                    'Mức độ hoàn chỉnh': completeness # MỚI: Thêm cột mới
                })

    if not audio_data:
        print("Không tìm thấy file âm thanh nào trong thư mục được chỉ định.")
        return None

    return pd.DataFrame(audio_data)

# --- CẤU HÌNH VÀ THỰC THI ---
if __name__ == "__main__":
    # THAY ĐỔI ĐƯỜNG DẪN NÀY
    main_folder_path = r'H:\Toàn-Khang_NCKH\ngtai_dataset\dataset_unhealthy'

    # Tên file Excel đầu ra
    output_excel_file = r'H:\Toàn-Khang_NCKH\ngtai_dataset\newmetadata_completed.xlsx'

    # Gọi hàm để phân tích
    df_results = analyze_audio_directory(main_folder_path)

    # Nếu có kết quả thì lưu ra file Excel
    if df_results is not None:
        try:
            # Sắp xếp lại các cột theo thứ tự mong muốn
            df_results = df_results[['Lớp', 'Tên file', 'Định dạng', 'Thời gian (giây)', 'Mức độ hoàn chỉnh']]
            
            # Xuất ra file Excel
            df_results.to_excel(output_excel_file, index=False)
            print(f"\n Hoàn tất! Đã lưu kết quả vào file: {output_excel_file}")

            # --- THỐNG KÊ BỔ SUNG ---
            print("\n--- THỐNG KÊ TỔNG QUAN ---")
            
            # 1. Số lượng file cho mỗi lớp
            print("\nSố lượng file cho mỗi lớp:")
            print(df_results['Lớp'].value_counts())
            
            # 2. MỚI: Thống kê mức độ hoàn chỉnh
            print("\nThống kê theo mức độ hoàn chỉnh:")
            print(df_results['Mức độ hoàn chỉnh'].value_counts())
            
            # 3. Tổng thời gian cho mỗi lớp
            print("\nTổng thời gian (phút) cho mỗi lớp (chỉ tính các file 'Hoàn chỉnh'):")
            # Chuyển cột thời gian sang kiểu số, lỗi sẽ được coi là NaN
            df_results['Thời gian (giây)'] = pd.to_numeric(df_results['Thời gian (giây)'], errors='coerce')
            
            # MỚI: Chỉ tính tổng thời gian cho các file không bị hỏng và không im lặng
            completed_files_df = df_results[df_results['Mức độ hoàn chỉnh'] == 'Hoàn chỉnh'].copy()
            
            # Nhóm theo lớp và tính tổng thời gian
            total_duration_per_class = completed_files_df.groupby('Lớp')['Thời gian (giây)'].sum() / 60
            print(total_duration_per_class.round(2))

        except Exception as e:
            print(f"LỖI: Không thể lưu file Excel. Lý do: {e}")