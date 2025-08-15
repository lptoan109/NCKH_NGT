# -*- coding: utf-8 -*-
"""
Đoạn mã chuyên nghiệp để giải nén các tập tin nén sử dụng 7-Zip.

Đoạn mã này được thiết kế để thực thi tự động mà không cần tương tác.
Nó tận dụng tốc độ và sự ổn định của công cụ dòng lệnh 7-Zip để thực hiện
việc giải nén, với các tính năng hỗ trợ tái tục (bằng cách bỏ qua các
tập tin đã tồn tại) và báo cáo lỗi chi tiết.

Tất cả các đường dẫn phải được cấu hình trong phần CẤU HÌNH trước khi chạy.
"""
import subprocess
import sys
from pathlib import Path

# --- CẤU HÌNH ---
# Người dùng cần chỉnh sửa các đường dẫn trong phần này.

# Đường dẫn ĐẦY ĐỦ đến file thực thi 7z.exe.
# Ví dụ: r"C:\Program Files\7-Zip\7z.exe"
SEVEN_ZIP_EXECUTABLE = Path(r"C:\Program Files\7-Zip\7z.exe")

# Đường dẫn ĐẦY ĐỦ đến file nén bạn muốn giải nén.
# Ví dụ: r"D:\TaiLieu\DuAnLon.zip"
SOURCE_ARCHIVE = Path(r"E:\Toàn-Khang_NCKH\ngtai_dataset\TBscreen_Dataset.zip")

# Đường dẫn ĐẦY ĐỦ đến thư mục bạn muốn chứa kết quả giải nén.
# Thư mục sẽ được tự động tạo nếu chưa tồn tại.
# Ví dụ: r"E:\KetQuaGiaiNen\DuAnLon"
DESTINATION_DIRECTORY = Path(r"F:\NCKH_Toàn_Khang\TBSCREENDATASET")
# --- KẾT THÚC CẤU HÌNH ---


def decompress_archive(
    seven_zip_path: Path, archive_path: Path, output_dir: Path
) -> bool:
    """
    Giải nén một tập tin nén bằng 7-Zip với khả năng tái tục và xử lý lỗi.

    Tham số:
        seven_zip_path: Đường dẫn đến file thực thi 7z.exe.
        archive_path: Đường dẫn đến tập tin nén nguồn.
        output_dir: Đường dẫn đến thư mục đích để giải nén.

    Trả về:
        True nếu giải nén thành công, ngược lại trả về False.
    """
    print("-" * 70)
    print(f"Bắt đầu quá trình giải nén...")
    print(f"  Nguồn: {archive_path}")
    print(f"  Đích:  {output_dir}")

    # 1. Kiểm tra tính hợp lệ của các đường dẫn
    if not seven_zip_path.is_file():
        print(f"\nLỖI: Không tìm thấy file thực thi 7-Zip tại: {seven_zip_path}")
        print("Vui lòng sửa lại.")
        return False

    if not archive_path.is_file():
        print(f"\nLỖI: Không tìm thấy tập tin nén nguồn tại: {archive_path}")
        print("Vui lòng sửa lại.")
        return False

    # 2. Tạo thư mục đích nếu chưa tồn tại
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"\nLỖI: Không thể tạo thư mục đích: {output_dir}")
        print(f"Lỗi hệ thống: {e}")
        return False

    # 3. Xây dựng lệnh 7-Zip
    #    'x'      : Giải nén với đường dẫn đầy đủ (giữ nguyên cấu trúc thư mục).
    #    '-o{dir}': Chỉ định thư mục đầu ra (viết liền, không có khoảng trắng).
    #    '-aos'   : Bỏ qua các tập tin đã tồn tại. Đây là chìa khóa cho tính năng "tái tục".
    #    '-y'     : Tự động trả lời "Yes" cho mọi câu hỏi từ 7-Zip.
    command = [
        str(seven_zip_path),
        "x",
        str(archive_path),
        f"-o{output_dir}",
        "-aos",
        "-y",
    ]

    # 4. Thực thi lệnh
    print("\nĐang thực thi lệnh 7-Zip...")
    try:
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="surrogateescape",
        )
    except FileNotFoundError:
        print(f"\nLỖI NGHIÊM TRỌNG: Không thể thực thi lệnh. Đường dẫn '{seven_zip_path}' đã chính xác chưa?")
        return False
    except Exception as e:
        print(f"\nLỖI NGHIÊM TRỌNG: Một lỗi bất ngờ đã xảy ra khi thực thi tiến trình con: {e}")
        return False

    # 5. Xử lý kết quả
    # In ra kết quả chuẩn từ 7-Zip, thường chứa danh sách các tệp được giải nén
    if process.stdout:
        print("\n--- Kết quả từ 7-Zip ---")
        print(process.stdout.strip())
        print("--------------------------")

    # Mã trả về khác 0 từ 7-Zip cho biết đã có lỗi xảy ra
    if process.returncode != 0:
        print("\nLỖI: 7-Zip báo cáo có lỗi trong quá trình giải nén.")
        print(f"Mã trả về: {process.returncode}")
        if process.stderr:
            print("\n--- Chi tiết lỗi từ 7-Zip ---")
            print(process.stderr.strip())
            print("--------------------------------")
            print("\nNguyên nhân thường gặp: Tập tin nén bị lỗi, sai mật khẩu, hoặc không đủ dung lượng đĩa.")
        return False

    print("\nTHÀNH CÔNG: Quá trình giải nén đã hoàn tất.")
    return True


def main():
    """Hàm chính để chạy kịch bản giải nén."""
    try:
        if decompress_archive(
            SEVEN_ZIP_EXECUTABLE, SOURCE_ARCHIVE, DESTINATION_DIRECTORY
        ):
            sys.exit(0)  # Thoát với mã thành công
        else:
            sys.exit(1)  # Thoát với mã lỗi
    except KeyboardInterrupt:
        print("\n\nQuá trình bị người dùng ngắt (Ctrl+C). Đang thoát.")
        sys.exit(1)


if __name__ == "__main__":
    main()