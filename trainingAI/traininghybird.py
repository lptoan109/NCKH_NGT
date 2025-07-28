################################################################################
# BƯỚC 1: THIẾT LẬP MÔI TRƯỜNG VÀ CÀI ĐẶT THƯ VIỆN
################################################################################
from google.colab import drive
drive.mount('/content/drive')

!pip install -q librosa tensorflow pandas scikit-learn matplotlib seaborn pytz xgboost tqdm shap

import os
import pandas as pd
import numpy as np
import librosa
import tensorflow as tf
import datetime
import pytz
import joblib
import xgboost as xgb
import shap
from tqdm.notebook import tqdm
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("TensorFlow Version:", tf.__version__)

################################################################################
# BƯỚC 2: CẤU HÌNH DỰ ÁN
################################################################################
print("\n--- Bắt đầu Bước 2: Cấu hình và chuẩn bị dữ liệu ---")

INPUT_FOLDER = '/content/ngtai_data/ngtai_dataset/'
OUTPUT_FOLDER = '/content/drive/MyDrive/Tai_Lieu_NCKH/newAiData'
USE_SEGMENTATION = True
DURATION, SEGMENT_DURATION, ENERGY_THRESHOLD_DB = 5, 2, 20

XGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}
EARLY_STOPPING_ROUNDS = 20

if not os.path.isdir(INPUT_FOLDER): raise ValueError("Lỗi: Đường dẫn INPUT_FOLDER không tồn tại.")
if not os.path.isdir(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

vn_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
timestamp = datetime.datetime.now(vn_timezone).strftime('%Y%m%d_%H%M%S')
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, f'output_{timestamp}')
os.makedirs(FINAL_OUTPUT_PATH, exist_ok=True)
print(f"Tất cả kết quả sẽ được lưu tại: {FINAL_OUTPUT_PATH}")

################################################################################
# BƯỚC 3: TẠO MANIFEST VÀ PHÂN CHIA DỮ LIỆU
################################################################################
print("\n--- Bắt đầu Bước 3: Tạo manifest từ cấu trúc thư mục mới ---")

POSITIVE_FOLDERS = [os.path.join(INPUT_FOLDER, 'cough'), os.path.join(INPUT_FOLDER, 'breathing')]
NEGATIVE_FOLDER = os.path.join(INPUT_FOLDER, 'noise')

def extract_patient_id(filepath):
    filename = os.path.basename(filepath)
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
print(f"Đã quét xong. Tổng số file hợp lệ: {len(df)}")

unique_patients = df['patient_id'].unique()
train_val_pids, test_pids = train_test_split(unique_patients, test_size=0.2, random_state=42)
train_pids, val_pids = train_test_split(train_val_pids, test_size=0.2, random_state=42)
train_df = df[df['patient_id'].isin(train_pids)]
val_df = df[df['patient_id'].isin(val_pids)]
test_df = df[df['patient_id'].isin(test_pids)]

################################################################################
# BƯỚC 4: XÂY DỰNG BỘ TRÍCH XUẤT ĐẶC TRƯNG VÀ XỬ LÝ DỮ LIỆU (THEO LÔ)
################################################################################
print("\n--- Bắt đầu Bước 4: Trích xuất đặc trưng theo từng lô ---")
SAMPLE_RATE, N_MELS, IMG_SIZE = 16000, 224, (224, 224)
FEATURE_EXTRACTION_BATCH_SIZE = 1000

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

def process_in_batches(df, name):
    temp_dir = os.path.join(FINAL_OUTPUT_PATH, f'temp_{name}')
    os.makedirs(temp_dir, exist_ok=True)
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
                print(f"Lỗi file {row['filepath']}: {e}")
        np.save(os.path.join(temp_dir, f'features_{batch_num}.npy'), np.array(features))
        np.save(os.path.join(temp_dir, f'labels_{batch_num}.npy'), np.array(labels))
        batch_num += 1
    
    all_features = np.concatenate([np.load(os.path.join(temp_dir, f)) for f in sorted(os.listdir(temp_dir)) if 'features' in f])
    all_labels = np.concatenate([np.load(os.path.join(temp_dir, f)) for f in sorted(os.listdir(temp_dir)) if 'labels' in f])
    shutil.rmtree(temp_dir)
    return all_features, all_labels

X_train, y_train = process_in_batches(train_df, "train")
X_val, y_val = process_in_batches(val_df, "val")
X_test, y_test = process_in_batches(test_df, "test")

################################################################################
# BƯỚC 5 & 6: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH (XGBOOST)
################################################################################
print("\n--- Bắt đầu Bước 5 & 6: Huấn luyện mô hình XGBoost ---")
classifier = xgb.XGBClassifier(**XGB_PARAMS)

# <<< SỬA LỖI: Dùng đúng cú pháp early_stopping_rounds cho scikit-learn wrapper >>>
classifier.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             early_stopping_rounds=EARLY_STOPPING_ROUNDS,
             verbose=False)

model_path = os.path.join(FINAL_OUTPUT_PATH, f'{timestamp}_xgboost_model.joblib')
joblib.dump(classifier, model_path)
print(f"Đã lưu mô hình XGBoost tại: {model_path}")

################################################################################
# BƯỚC 7: ĐÁNH GIÁ VÀ GIẢI THÍCH MÔ HÌNH
################################################################################
print("\n--- Bắt đầu Bước 7: Đánh giá và Giải thích mô hình ---")
y_pred = classifier.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Độ chính xác cuối cùng trên tập thử nghiệm: {accuracy * 100:.2f}%")

print("\nBáo cáo Phân loại:")
report = classification_report(y_test, y_pred, target_names=['Noise', 'Cough/Breathing'])
print(report)
report_path = os.path.join(FINAL_OUTPUT_PATH, f'{timestamp}_report.txt')
with open(report_path, 'w') as f:
    f.write(f"Cấu hình XGBoost:\n{XGB_PARAMS}\n")
    f.write(f"Độ chính xác: {accuracy * 100:.2f}%\n\n{report}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Noise', 'Cough/Breathing'], yticklabels=['Noise', 'Cough/Breathing'])
plt.xlabel('Dự đoán'); plt.ylabel('Thực tế'); plt.title('Ma trận nhầm lẫn')
plt.savefig(os.path.join(FINAL_OUTPUT_PATH, f'{timestamp}_confusion_matrix.png'))
plt.show()

print("\n--- Bắt đầu tính toán và tạo biểu đồ SHAP ---")
explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_test)

plt.figure()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=30)
plt.title(f'SHAP - Global Feature Importance', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FINAL_OUTPUT_PATH, f'{timestamp}_shap_summary_bar.png'))
plt.show()

shap.initjs()
force_plot = shap.force_plot(explainer.expected_value, shap_values[0,:], X_test[0,:], show=False)
shap.save_html(os.path.join(FINAL_OUTPUT_PATH, f'{timestamp}_shap_force_plot.html'), force_plot)
print(f"Đã lưu Force Plot dưới dạng file HTML.")
display(force_plot)

print(f"\n--- Pipeline đã hoàn thành! Mọi kết quả đã được lưu tại {FINAL_OUTPUT_PATH} ---")