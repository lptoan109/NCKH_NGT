import os
import pandas as pd
from pydub import AudioSegment
from pydub.utils import mediainfo

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
    supported_formats = ['.wav']

    print(f"Bắt đầu quét thư mục: {root_folder}\n...")

    # os.walk sẽ duyệt qua tất cả các thư mục và file một cách đệ quy
    for dirpath, _, filenames in os.walk(root_folder):
        # Tên lớp chính là tên của thư mục con
        class_name = os.path.basename(dirpath)

        for filename in filenames:
            # Tách tên file và đuôi file để kiểm tra định dạng
            file_ext = os.path.splitext(filename)[1].lower()

            if file_ext in supported_formats:
                file_path = os.path.join(dirpath, filename)
                try:
                    # Dùng mediainfo để lấy thời lượng, nhanh hơn là load cả file
                    info = mediainfo(file_path)
                    duration_seconds = float(info['duration'])

                    # Thu thập dữ liệu
                    audio_data.append({
                        'Lớp': class_name,
                        'Tên file': filename,
                        'Định dạng': file_ext,
                        'Thời gian (giây)': round(duration_seconds, 2) # Làm tròn 2 chữ số
                    })
                    print(f"Đã xử lý: {file_path}")

                except Exception as e:
                    print(f"LỖI: Không thể xử lý file {file_path}. Lý do: {e}")
                    # Ghi nhận file lỗi nếu muốn
                    audio_data.append({
                        'Lớp': class_name,
                        'Tên file': filename,
                        'Định dạng': file_ext,
                        'Thời gian (giây)': 'Lỗi xử lý'
                    })

    if not audio_data:
        print("Không tìm thấy file âm thanh nào trong thư mục được chỉ định.")
        return None

    return pd.DataFrame(audio_data)

# --- CẤU HÌNH VÀ THỰC THI ---
if __name__ == "__main__":
    # THAY ĐỔI ĐƯỜNG DẪN NÀY
    # Thay 'D:/path/to/your/audio_folder' bằng đường dẫn thực tế của bạn
    main_folder_path = r'H:\Toàn-Khang_NCKH\ngtai_dataset\dataset_unhealthy'

    # Tên file Excel đầu ra
    output_excel_file = r'H:\Toàn-Khang_NCKH\ngtai_dataset\newmetadata.xlsx'

    # Gọi hàm để phân tích
    df_results = analyze_audio_directory(main_folder_path)

    # Nếu có kết quả thì lưu ra file Excel
    if df_results is not None:
        try:
            # Sắp xếp lại các cột theo thứ tự mong muốn
            df_results = df_results[['Lớp', 'Tên file', 'Định dạng', 'Thời gian (giây)']]
            
            # Xuất ra file Excel
            df_results.to_excel(output_excel_file, index=False)
            print(f"\n Hoàn tất! Đã lưu kết quả vào file: {output_excel_file}")

            # Thống kê bổ sung
            print("\n--- THỐNG KÊ TỔNG QUAN ---")
            print("Số lượng file cho mỗi lớp:")
            print(df_results['Lớp'].value_counts())
            
            print("\nTổng thời gian (phút) cho mỗi lớp:")
            # Chuyển cột thời gian sang kiểu số, lỗi sẽ được coi là NaN
            df_results['Thời gian (giây)'] = pd.to_numeric(df_results['Thời gian (giây)'], errors='coerce')
            # Nhóm theo lớp và tính tổng thời gian
            total_duration_per_class = df_results.groupby('Lớp')['Thời gian (giây)'].sum() / 60
            print(total_duration_per_class.round(2))


        except Exception as e:
            print(f"LỖI: Không thể lưu file Excel. Lý do: {e}")