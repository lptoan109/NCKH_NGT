# -*- coding: utf-8 -*-
"""
Kịch bản chuyên nghiệp để giải nén các tập tin nén sử dụng 7-Zip.

Kịch bản này được thiết kế để thực thi tự động mà không cần tương tác.
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
# Sử dụng chuỗi r'' để tránh các vấn đề với dấu gạch chéo ngược trong Windows.

# Đường dẫn ĐẦY ĐỦ đến file thực thi 7z.exe.
# Ví dụ: r"C:\Program Files\7-Zip\7z.exe"
SEVEN_ZIP_EXECUTABLE = Path(r"C:\Program Files\7-Zip\7z.exe")

# Đường dẫn ĐẦY ĐỦ đến file nén bạn muốn giải nén.
SOURCE_ARCHIVE = Path(r"E:\Toàn-Khang_NCKH\ngtai_dataset\TBscreen_Dataset.zip")

# Đường dẫn ĐẦY ĐỦ đến thư mục bạn muốn chứa kết quả giải nén.
# Thư mục sẽ được tự động tạo nếu chưa tồn tại.
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
        print("Vui lòng sửa lại đường dẫn SEVEN_ZIP_EXECUTABLE trong kịch bản.")
        return False

    if not archive_path.is_file():
        print(f"\nLỖI: Không tìm thấy tập tin nén nguồn tại: {archive_path}")
        print("Vui lòng sửa lại đường dẫn SOURCE_ARCHIVE trong kịch bản.")
        return False

    # 2. Tạo thư mục đích nếu chưa tồn tại
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"\nLỖI: Không thể tạo thư mục đích: {output_dir}")
        print(f"Lỗi hệ thống: {e}")
        return False

    # 3. Xây dựng lệnh 7-Zip
    command = [
        str(seven_zip_path), "x", str(archive_path), f"-o{output_dir}", "-aos", "-y",
    ]

    # 4. Thực thi lệnh và xử lý output theo thời gian thực
    print("\nĐang thực thi lệnh 7-Zip...")
    print("--- Log giải nén thời gian thực ---")
    try:
        # Sử dụng Popen để chạy tiến trình và đọc output đồng thời
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="surrogateescape",
            bufsize=1,  # Đảm bảo output được gửi theo từng dòng
        )

        # Đọc và in từng dòng output ngay khi có
        if process.stdout:
            for line in iter(process.stdout.readline, ""):
                cleaned_line = line.strip()
                # 7-Zip thường in ra các dòng thông tin như "Scanning the drive for archives"
                # hoặc dòng trống. Ta chỉ hiển thị các dòng có nội dung.
                if cleaned_line and not cleaned_line.startswith("7-Zip"):
                    print(f"  [HOÀN THÀNH] {cleaned_line}")
        
        # Đợi tiến trình kết thúc và lấy kết quả cuối cùng
        return_code = process.wait()
        stderr_output = process.stderr.read()

    except FileNotFoundError:
        print(f"\nLỖI NGHIÊM TRỌNG: Không thể thực thi lệnh. Đường dẫn '{seven_zip_path}' đã chính xác chưa?")
        return False
    except Exception as e:
        print(f"\nLỖI NGHIÊM TRỌNG: Một lỗi bất ngờ đã xảy ra khi thực thi tiến trình con: {e}")
        return False

    # 5. Xử lý kết quả cuối cùng
    print("------------------------------------")
    if return_code != 0:
        print("\nLỖI: 7-Zip báo cáo có lỗi trong quá trình giải nén.")
        print(f"Mã trả về: {return_code}")
        if stderr_output:
            print("\n--- Chi tiết lỗi từ 7-Zip ---")
            print(stderr_output.strip())
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