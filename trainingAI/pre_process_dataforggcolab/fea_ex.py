# ======================================================================================
# GIAI ĐOẠN 2: TRÍCH XUẤT ĐẶC TRƯNG (PHIÊN BẢN CUỐI CÙNG - TẢI LÊN THEO LÔ)
# ======================================================================================
print("GIAI ĐOẠN 2: TRÍCH XUẤT ĐẶC TRƯNG...")

# --- CÀI ĐẶT VÀ IMPORT ---
!pip install kaggle librosa tqdm pandas torch torchaudio noisereduce -q
import os
import json
import librosa
import numpy as np
from tqdm.auto import tqdm
import sys
import torch
import torchaudio.transforms as T
import noisereduce as nr
from google.colab import drive
import subprocess
import shutil

# --- CẤU HÌNH ---
class CONFIG:
    KAGGLE_USERNAME = "lptoan"
    RAW_DATASET_ID = "raw-audio-dataset"
    PROCESSED_FEATURES_DATASET_ID = "processed-features-dataset"
    
    # !!! THAM SỐ QUAN TRỌNG: Kích thước của mỗi lô tải lên
    # Sau khi xử lý 500 file, script sẽ tải lên Kaggle 1 lần.
    # Bạn có thể tăng/giảm con số này. 500 là một lựa chọn tốt.
    BATCH_SIZE_FOR_UPLOAD = 500

    KAGGLE_JSON_DRIVE_PATH = "/content/drive/MyDrive/Tai_Lieu_NCKH/KaggleAPI/kaggle.json"
    DRIVE_MOUNT_PATH = "/content/drive"
    VALID_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".m4a")

# --- Cấu hình cho Mel Spectrogram ---
SAMPLE_RATE = 16000
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

# --- KẾT NỐI, XÁC THỰC VÀ THIẾT LẬP THƯ MỤC ---
drive.mount(CONFIG.DRIVE_MOUNT_PATH, force_remount=True)
!mkdir -p ~/.kaggle
!cp "{CONFIG.KAGGLE_JSON_DRIVE_PATH}" ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json
print("Xác thực Kaggle API thành công!")

RAW_AUDIO_LOCAL_PATH = "downloaded_raw_audio"
FEATURES_BATCH_PATH = "features_for_current_batch" # Thư mục chứa lô hiện tại
os.makedirs(RAW_AUDIO_LOCAL_PATH, exist_ok=True)
os.makedirs(FEATURES_BATCH_PATH, exist_ok=True)

# --- KHỞI TẠO BỘ TRÍCH XUẤT ĐẶC TRƯNG ---
mel_spectrogram_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
)

# --- LẤY DANH SÁCH FILE ĐÃ XỬ LÝ ---
print(f"Kiểm tra các file đã tồn tại trong dataset '{CONFIG.PROCESSED_FEATURES_DATASET_ID}'...")
try:
    processed_files_output = subprocess.check_output(f'kaggle datasets files {CONFIG.KAGGLE_USERNAME}/{CONFIG.PROCESSED_FEATURES_DATASET_ID}', shell=True)
    processed_files_list = processed_files_output.decode('utf-8').strip().split('\n')[2:]
    processed_basenames = {os.path.basename(f).replace('.npy', '') for f in processed_files_list}
    print(f"Đã tìm thấy {len(processed_basenames)} file đặc trưng đã được xử lý.")
except subprocess.CalledProcessError:
    processed_basenames = set()
    print("Dataset đặc trưng còn trống. Bắt đầu từ đầu.")

# --- TẢI DATASET ÂM THANH THÔ ---
print(f"\nBắt đầu tải dataset âm thanh thô '{CONFIG.RAW_DATASET_ID}'...")
!kaggle datasets download {CONFIG.KAGGLE_USERNAME}/{CONFIG.RAW_DATASET_ID} -p {RAW_AUDIO_LOCAL_PATH} --unzip
print("Tải và giải nén hoàn tất.")

# --- TÌM VÀ XỬ LÝ CÁC FILE CÒN LẠI ---
audio_files_to_process = []
for root, dirs, files in os.walk(RAW_AUDIO_LOCAL_PATH):
    for file in files:
        if file.endswith(CONFIG.VALID_AUDIO_EXTENSIONS):
            audio_files_to_process.append(os.path.join(root, file))

if not audio_files_to_process:
    print("LỖI: Không tìm thấy file âm thanh nào.")
else:
    batch_counter = 0
    total_files = len(audio_files_to_process)
    print(f"Tổng cộng có {total_files} file âm thanh. Bắt đầu xử lý...")
    
    for i, audio_path in enumerate(tqdm(audio_files_to_process, desc="Tổng tiến độ")):
        try:
            relative_path = os.path.relpath(audio_path, RAW_AUDIO_LOCAL_PATH)
            base_name = os.path.splitext(relative_path)[0].replace(os.sep, '_')
            
            if base_name in processed_basenames:
                continue
            
            y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
            y_clean = nr.reduce_noise(y=y, sr=sr)
            audio_tensor = torch.tensor(y_clean, dtype=torch.float32)
            mel_spectrogram = mel_spectrogram_transform(audio_tensor)
            log_mel_spectrogram = T.AmplitudeToDB()(mel_spectrogram)
            spectrogram_numpy = log_mel_spectrogram.squeeze().numpy()
            
            output_path = os.path.join(FEATURES_BATCH_PATH, f"{base_name}.npy")
            np.save(output_path, spectrogram_numpy)
            batch_counter += 1

            # !!! LOGIC QUAN TRỌNG: KIỂM TRA ĐỂ TẢI LÊN LÔ !!!
            # Tải lên nếu đủ một lô HOẶC nếu đây là file cuối cùng
            if batch_counter >= CONFIG.BATCH_SIZE_FOR_UPLOAD or (i + 1) == total_files:
                if batch_counter > 0: # Chỉ tải lên nếu có file mới
                    print(f"\nĐã đủ một lô {batch_counter} file. Bắt đầu tải lên Kaggle...")
                    !kaggle datasets version -p "{FEATURES_BATCH_PATH}" -m "Add batch of {batch_counter} new features"
                    print("Tải lên lô thành công!")
                    
                    # Dọn dẹp thư mục lô để chuẩn bị cho lần tiếp theo
                    batch_counter = 0
                    for filename in os.listdir(FEATURES_BATCH_PATH):
                        os.remove(os.path.join(FEATURES_BATCH_PATH, filename))
        
        except Exception as e:
            print(f"Lỗi khi xử lý file {audio_path}: {e}")
            
    print("\nKhông còn file mới nào để xử lý. Mọi thứ đã hoàn tất!")
    print(f"\nHOÀN TẤT GIAI ĐOẠN 2!")