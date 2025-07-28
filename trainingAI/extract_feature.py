# -*- coding: utf-8 -*-
"""
Script 1: Tải, Giải nén, Trích xuất Đặc trưng và Lưu trữ Vĩnh viễn.
Chỉ cần chạy script này một lần duy nhất.
"""

################################################################################
# BƯỚC 1: CÀI ĐẶT VÀ IMPORT THƯ VIỆN
################################################################################
print("STEP 1: Cài đặt và import thư viện...")
!pip install -q gdown
!apt-get update -qq && apt-get install -y p7zip-full -qq
!pip install -q librosa tensorflow pandas scikit-learn matplotlib seaborn pytz tqdm openpyxl

from google.colab import drive
import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
import datetime
import pytz
import shutil
import gdown
import glob
from tqdm.notebook import tqdm
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

print("STEP 1: Hoàn tất.")

################################################################################
# BƯỚC 2: KẾT NỐI VỚI GOOGLE DRIVE
################################################################################
print("\nSTEP 2: Kết nối với Google Drive...")
drive.mount('/content/drive')
print("-> Kết nối thành công.")

################################################################################
# BƯỚC 3: CẤU HÌNH
################################################################################
print("\nSTEP 3: Cấu hình...")
# --- ⚙️ 3.1 CẤU HÌNH TẢI DỮ LIỆU ---
FOLDER_ID = '19OcHdaHqglR75HoKgIUvUFWcKUWwPP3A'
MAIN_SPLIT_FILENAME = 'ngtai_dataset.zip.111.001'
WORKING_DIRECTORY = '/content/ngtai_data/'

# --- ⚙️ 3.2 CẤU HÌNH LƯU TRỮ VÀ XỬ LÝ ---
FEATURE_STORAGE_FOLDER = '/content/drive/MyDrive/Tai_Lieu_NCKH/PrecomputedFeatures/'
USE_SEGMENTATION = True
DURATION, SEGMENT_DURATION, ENERGY_THRESHOLD_DB = 5, 2, 20
FEATURE_EXTRACTION_BATCH_SIZE = 1000
SAMPLE_RATE, N_MELS, IMG_SIZE = 16000, 224, (224, 224)
# ------------------------------------
print("-> Cấu hình hoàn tất.")

os.makedirs(WORKING_DIRECTORY, exist_ok=True)
os.makedirs(FEATURE_STORAGE_FOLDER, exist_ok=True)
print(f"Các file đặc trưng sẽ được lưu tại: {FEATURE_STORAGE_FOLDER}")

################################################################################
# BƯỚC 4 (TỐI ƯU HÓA): TẢI VÀ GIẢI NÉN TRỰC TIẾP
################################################################################
print(f"\nSTEP 4: Tải dữ liệu từ Google Drive Folder ID: {FOLDER_ID}...")
gdown.download_folder(id=FOLDER_ID, output=WORKING_DIRECTORY, quiet=False, use_cookies=False)

print("\n-> Bắt đầu quá trình giải nén trực tiếp...")
search_pattern = os.path.join(WORKING_DIRECTORY, '**', MAIN_SPLIT_FILENAME)
found_files = glob.glob(search_pattern, recursive=True)

if not found_files:
    raise FileNotFoundError(f"Lỗi: Không tìm thấy file chính '{MAIN_SPLIT_FILENAME}' trong '{WORKING_DIRECTORY}'.")

main_split_filepath = found_files[0]
print(f"-> Tìm thấy file chính tại: {main_split_filepath}. Bắt đầu giải nén...")

# Trực tiếp giải nén vào thư mục làm việc chính (WORKING_DIRECTORY)
# 7z sẽ tự động tìm các file .002, .003,... để ghép và giải nén.
!7z x "{main_split_filepath}" -o"{WORKING_DIRECTORY}" -y
print("-> Giải nén hoàn tất.")

# Kiểm tra xem thư mục dữ liệu cuối cùng có tồn tại không
INPUT_FOLDER = os.path.join(WORKING_DIRECTORY, 'ngtai_dataset')
if not os.path.isdir(INPUT_FOLDER):
    raise NotADirectoryError(f"Lỗi: Thư mục dữ liệu '{INPUT_FOLDER}' không tồn tại sau khi giải nén. Vui lòng kiểm tra lại tên thư mục bên trong file nén.")

################################################################################
# BƯỚC 5: TẠO MANIFEST VÀ PHÂN CHIA DỮ LIỆU
################################################################################
print("\nSTEP 5: Tạo manifest và phân chia dữ liệu...")
POSITIVE_FOLDERS = [os.path.join(INPUT_FOLDER, 'cough'), os.path.join(INPUT_FOLDER, 'breathing')]
NEGATIVE_FOLDER = os.path.join(INPUT_FOLDER, 'noise')

def extract_patient_id(filename):
    parts = filename.split('_')
    return parts[0] if len(parts) > 1 else os.path.splitext(filename)[0]

data = []
for folder in POSITIVE_FOLDERS:
    if os.path.isdir(folder):
        for root, _, files in os.walk(folder):
            for filename in files:
                if filename.lower().endswith(('.wav', '.mp3', '.flac')):
                    data.append({'filepath': os.path.join(root, filename), 'label': 1, 'patient_id': extract_patient_id(filename)})
if os.path.isdir(NEGATIVE_FOLDER):
    for filename in os.listdir(NEGATIVE_FOLDER):
        if filename.lower().endswith(('.wav', '.mp3', '.flac')):
            data.append({'filepath': os.path.join(NEGATIVE_FOLDER, filename), 'label': 0, 'patient_id': os.path.splitext(filename)[0]})

df = pd.DataFrame(data).sample(frac=1).reset_index(drop=True)

unique_patients = df['patient_id'].unique()
train_val_pids, test_pids = train_test_split(unique_patients, test_size=0.2, random_state=42)
train_pids, val_pids = train_test_split(train_val_pids, test_size=0.2, random_state=42)
train_df = df[df['patient_id'].isin(train_pids)]
val_df = df[df['patient_id'].isin(val_pids)]
test_df = df[df['patient_id'].isin(test_pids)]
print("-> Phân chia dữ liệu hoàn tất.")

################################################################################
# BƯỚC 6: TRÍCH XUẤT ĐẶC TRƯNG VÀ LƯU TRỮ
################################################################################
print("\nSTEP 6: Trích xuất đặc trưng...")

def build_feature_extractor(input_shape):
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    inputs = Input(shape=input_shape)
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    outputs = GlobalAveragePooling2D()(x)
    return Model(inputs, outputs)

feature_extractor = build_feature_extractor((*IMG_SIZE, 3))
FEATURE_DIM = feature_extractor.output.shape[1]

def segment_cough(signal, sr, top_db):
    intervals = librosa.effects.split(signal, top_db=top_db)
    if not np.any(intervals): return signal
    max_energy, best_interval = 0, intervals[0]
    for start, end in intervals:
        energy = np.sum(signal[start:end]**2)
        if energy > max_energy: max_energy, best_interval = energy, (start, end)
    return signal[best_interval[0]:best_interval[1]]

def extract_features_for_cough(filepath):
    signal, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
    if USE_SEGMENTATION: signal = segment_cough(signal, sr, top_db=ENERGY_THRESHOLD_DB)
    target_length = int(SEGMENT_DURATION * sr)
    if len(signal) < target_length: signal = np.pad(signal, (0, target_length - len(signal)), 'constant')
    else: signal = signal[:target_length]
    mel_spec = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=N_MELS)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    log_mel_spec_norm = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-6)
    resized_spec = tf.image.resize(np.expand_dims(log_mel_spec_norm, -1), IMG_SIZE)
    spec_rgb = tf.image.grayscale_to_rgb(resized_spec)
    return feature_extractor.predict(np.expand_dims(spec_rgb, 0), verbose=0)[0]

def extract_features_for_no_cough(filepath):
    signal, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
    target_length = int((SEGMENT_DURATION if USE_SEGMENTATION else DURATION) * sr)
    if len(signal) > target_length:
        start = np.random.randint(0, len(signal) - target_length + 1)
        signal = signal[start : start + target_length]
    elif len(signal) < target_length:
        signal = np.pad(signal, (0, target_length - len(signal)), 'constant')
    mfccs = np.mean(librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=40).T, axis=0)
    return mfccs

def process_and_save_features(df, name):
    temp_dir = os.path.join(FEATURE_STORAGE_FOLDER, f'temp_{name}')
    os.makedirs(temp_dir, exist_ok=True)
    error_logs = []
    batch_num = 0
    for i in tqdm(range(0, len(df), FEATURE_EXTRACTION_BATCH_SIZE), desc=f'Processing {name} set in batches'):
        batch_df = df.iloc[i:i + FEATURE_EXTRACTION_BATCH_SIZE]
        features, labels = [], []
        for _, row in batch_df.iterrows():
            try:
                if row['label'] == 1:
                    feature_vector = extract_features_for_cough(row['filepath'])
                else:
                    feature_vector_simple = extract_features_for_no_cough(row['filepath'])
                    feature_vector = np.zeros(FEATURE_DIM)
                    len_simple = len(feature_vector_simple)
                    feature_vector[:min(len_simple, FEATURE_DIM)] = feature_vector_simple[:FEATURE_DIM]
                features.append(feature_vector)
                labels.append(row['label'])
            except Exception as e:
                error_logs.append({'filepath': row['filepath'], 'error': str(e)})
        if features:
            np.save(os.path.join(temp_dir, f'features_{batch_num}.npy'), np.array(features))
            np.save(os.path.join(temp_dir, f'labels_{batch_num}.npy'), np.array(labels))
            batch_num += 1
    if error_logs:
        pd.DataFrame(error_logs).to_excel(os.path.join(FEATURE_STORAGE_FOLDER, 'error_log.xlsx'), index=False)
        print(f"-> Đã lưu danh sách {len(error_logs)} file lỗi tại file excel.")
    
    all_features = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(temp_dir, 'features_*.npy')))])
    all_labels = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(temp_dir, 'labels_*.npy')))])
    
    np.save(os.path.join(FEATURE_STORAGE_FOLDER, f'{name}_features.npy'), all_features)
    np.save(os.path.join(FEATURE_STORAGE_FOLDER, f'{name}_labels.npy'), all_labels)
    print(f"-> Đã lưu vĩnh viễn {name}_features.npy và {name}_labels.npy")
    
    shutil.rmtree(temp_dir)
    return all_features, all_labels

process_and_save_features(train_df, "train")
process_and_save_features(val_df, "val")
process_and_save_features(test_df, "test")

print("\n--- QUÁ TRÌNH TRÍCH XUẤT ĐẶC TRƯNG HOÀN TẤT ---")