################################################################################
# BƯỚC 1: THIẾT LẬP MÔI TRƯỜNG VÀ CÀI ĐẶT THƯ VIỆN
################################################################################
from google.colab import drive
drive.mount('/content/drive')

!pip install -q librosa tensorflow pandas scikit-learn matplotlib seaborn pytz

import os
import pandas as pd
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
from tensorflow.keras.applications import ResNet50V2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import pytz

print("TensorFlow Version:", tf.__version__)

################################################################################
# BƯỚC 2: CẤU HÌNH DỰ ÁN VÀ CHUẨN BỊ DỮ LIỆU
################################################################################
print("\n--- Bắt đầu Bước 2: Cấu hình và chuẩn bị dữ liệu ---")
INPUT_FOLDER = '/content/drive/MyDrive/DuLieuTiengHo/'
OUTPUT_FOLDER = '/content/drive/MyDrive/KetQuaNghienCuu/'
EPOCHS = 25
BATCH_SIZE = 32
PATIENCE = 5
USE_SEGMENTATION = True
DURATION = 5
SEGMENT_DURATION = 2
ENERGY_THRESHOLD_DB = 20

if not os.path.isdir(INPUT_FOLDER): raise ValueError("Lỗi: Đường dẫn INPUT_FOLDER không tồn tại.")
if not os.path.isdir(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

vn_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
timestamp = datetime.datetime.now(vn_timezone).strftime('%Y%m%d_%H%M%S')
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, f'output_{timestamp}')
os.makedirs(FINAL_OUTPUT_PATH, exist_ok=True)
print(f"Tất cả kết quả sẽ được lưu tại: {FINAL_OUTPUT_PATH}")

COUGH_PATH = os.path.join(INPUT_FOLDER, 'ho_va_tho')
NO_COUGH_PATH = os.path.join(INPUT_FOLDER, 'khong_ho_va_tho')
MANIFEST_CSV_PATH = os.path.join(FINAL_OUTPUT_PATH, f'{timestamp}_manifest.csv')

def extract_patient_id(filename): return filename.split('_')[0]

print("Đang tạo file manifest...")
data = [{'filepath': os.path.join(COUGH_PATH, f), 'label': 1, 'patient_id': extract_patient_id(f)} for f in os.listdir(COUGH_PATH) if f.endswith(('.wav', '.mp3', '.flac'))]
data.extend([{'filepath': os.path.join(NO_COUGH_PATH, f), 'label': 0, 'patient_id': extract_patient_id(f)} for f in os.listdir(NO_COUGH_PATH) if f.endswith(('.wav', '.mp3', '.flac'))])
df = pd.DataFrame(data)
df.to_csv(MANIFEST_CSV_PATH, index=False)
print(f"Đã tạo và lưu file manifest tại: {MANIFEST_CSV_PATH}")
print(f"Tổng số file: {len(df)}")
print(f"Tổng số bệnh nhân: {df['patient_id'].nunique()}")

################################################################################
# BƯỚC 3: PHÂN CHIA DỮ LIỆU THEO BỆNH NHÂN
################################################################################
print("\n--- Bắt đầu Bước 3: Phân chia dữ liệu theo bệnh nhân ---")
unique_patients = df['patient_id'].unique()
train_val_pids, test_pids = train_test_split(unique_patients, test_size=0.2, random_state=42)
train_pids, val_pids = train_test_split(train_val_pids, test_size=0.2, random_state=42)
train_df = df[df['patient_id'].isin(train_pids)]
val_df = df[df['patient_id'].isin(val_pids)]
test_df = df[df['patient_id'].isin(test_pids)]
print(f"Bệnh nhân trong tập huấn luyện: {len(train_pids)}, file: {len(train_df)}")
print(f"Bệnh nhân trong tập kiểm định: {len(val_pids)}, file: {len(val_df)}")
print(f"Bệnh nhân trong tập thử nghiệm: {len(test_pids)}, file: {len(test_df)}")

################################################################################
# BƯỚC 4: TIỀN XỬ LÝ VÀ TRÍCH XUẤT ĐẶC TRƯNG
################################################################################
print("\n--- Bắt đầu Bước 4: Thiết lập pipeline xử lý dữ liệu ---")
SAMPLE_RATE = 16000
N_MELS = 224
IMG_SIZE = (224, 224)

def segment_cough(signal, sr, top_db):
    intervals = librosa.effects.split(signal, top_db=top_db)
    if not intervals.any():
        return signal
    max_energy = 0
    best_interval = intervals[0]
    for start, end in intervals:
        energy = np.sum(signal[start:end]**2)
        if energy > max_energy:
            max_energy = energy
            best_interval = (start, end)
    return signal[best_interval[0]:best_interval[1]]

def process_audio_file(filepath, label):
    try:
        signal, sr = librosa.load(filepath, sr=SAMPLE_RATE, duration=DURATION)
        final_signal = None
        if label == 1 and USE_SEGMENTATION:
            final_signal = segment_cough(signal, sr, top_db=ENERGY_THRESHOLD_DB)
            target_duration = SEGMENT_DURATION
        else:
            target_duration = SEGMENT_DURATION if USE_SEGMENTATION else DURATION
            target_length_process = int(target_duration * sr)
            if len(signal) > target_length_process:
                start = np.random.randint(0, len(signal) - target_length_process + 1)
                final_signal = signal[start : start + target_length_process]
            else:
                final_signal = signal
        target_length = int(target_duration * sr)
        if len(final_signal) < target_length:
            final_signal = np.pad(final_signal, (0, target_length - len(final_signal)), 'constant')
        else:
            final_signal = final_signal[:target_length]
        mel_spec = librosa.feature.melspectrogram(y=final_signal, sr=sr, n_mels=N_MELS)
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        log_mel_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min() + 1e-6)
        resized_spec = tf.image.resize(np.expand_dims(log_mel_spec, -1), IMG_SIZE)
        resized_spec_rgb = tf.image.grayscale_to_rgb(resized_spec)
        return resized_spec_rgb, np.int64(label)
    except Exception as e:
        print(f"Lỗi xử lý file {filepath}: {e}")
        return tf.zeros((*IMG_SIZE, 3), dtype=tf.float32), np.int64(-1)

def create_tf_dataset(df):
    dataset = tf.data.Dataset.from_tensor_slices((df['filepath'].values, df['label'].values))
    dataset = dataset.map(lambda x, y: tf.py_function(process_audio_file, [x, y], [tf.float32, tf.int64]), num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.filter(lambda x, y: y != -1)

train_ds = create_tf_dataset(train_df).shuffle(1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = create_tf_dataset(val_df).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = create_tf_dataset(test_df).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
print("Đã tạo xong các pipeline dữ liệu cho TensorFlow với logic xử lý khác biệt.")

################################################################################
# BƯỚC 5: XÂY DỰNG MÔ HÌNH (SỬ DỤNG RESNET50V2)
################################################################################
print("\n--- Bắt đầu Bước 5: Xây dựng mô hình ---")
def build_model(input_shape):
    base_model = ResNet50V2(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False
    inputs = Input(shape=input_shape)
    x = tf.keras.applications.resnet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(1, activation='sigmoid')(x)
    return Model(inputs, outputs)

input_shape = (*IMG_SIZE, 3)
model = build_model(input_shape)
model.summary()

################################################################################
# BƯỚC 6: HUẤN LUYỆN MÔ HÌNH
################################################################################
print("\n--- Bắt đầu Bước 6: Huấn luyện mô hình ---")
model_checkpoint_path = os.path.join(FINAL_OUTPUT_PATH, f'{timestamp}_best_model.keras')
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, save_best_only=True, monitor='val_accuracy', mode='max'),
    tf.keras.callbacks.EarlyStopping(patience=PATIENCE, monitor='val_accuracy', restore_best_weights=True, mode='max')
]
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_ds, epochs=EPOCHS, validation_data=val_ds, callbacks=callbacks)

print("\n--- HOÀN THÀNH HUẤN LUYỆN. CHUYỂN SANG Ô TIẾP THEO ĐỂ ĐÁNH GIÁ. ---")

################################################################################
# BƯỚC 7: ĐÁNH GIÁ MÔ HÌNH VÀ LƯU KẾT QUẢ
################################################################################
print("\n--- Bắt đầu Bước 7: Đánh giá và Giải thích mô hình ---")
print("Tải lại mô hình tốt nhất từ checkpoint...")
try:
    model = tf.keras.models.load_model(model_checkpoint_path)
except NameError:
    print("Lỗi: Vui lòng chạy Ô 1 trước để huấn luyện và lưu mô hình.")
    raise

loss, accuracy = model.evaluate(test_ds)
print(f"Độ chính xác cuối cùng trên tập thử nghiệm: {accuracy * 100:.2f}%")

y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)
y_pred_probs = model.predict(test_ds)
y_pred = (y_pred_probs > 0.5).astype(int).flatten()

print("\nBáo cáo Phân loại:")
report = classification_report(y_true, y_pred, target_names=['Không Ho', 'Ho'])
print(report)
report_path = os.path.join(FINAL_OUTPUT_PATH, f'{timestamp}_report.txt')
with open(report_path, 'w') as f:
    f.write(f"Cấu hình: EPOCHS={EPOCHS}, BATCH_SIZE={BATCH_SIZE}, PATIENCE={PATIENCE}\n")
    f.write(f"Sử dụng phân đoạn: {USE_SEGMENTATION}\n")
    f.write("-" * 50 + "\n")
    f.write(f"Độ chính xác trên tập thử nghiệm: {accuracy * 100:.2f}%\n\n")
    f.write(report)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Không Ho', 'Ho'], yticklabels=['Không Ho', 'Ho'])
plt.xlabel('Dự đoán'); plt.ylabel('Thực tế'); plt.title('Ma trận nhầm lẫn')
plt.savefig(os.path.join(FINAL_OUTPUT_PATH, f'{timestamp}_confusion_matrix.png'))
plt.show()

pd.DataFrame(history.history).plot(figsize=(10, 6))
plt.grid(True); plt.gca().set_ylim(0, 1); plt.title('Lịch sử Huấn luyện'); plt.xlabel('Epochs')
plt.savefig(os.path.join(FINAL_OUTPUT_PATH, f'{timestamp}_training_history.png'))
plt.show()

################################################################################
# BƯỚC 8: GIẢI THÍCH MÔ HÌNH VỚI GRAD-CAM TRUNG BÌNH
################################################################################
print("\n--- Bắt đầu tính toán Grad-CAM trung bình cho lớp 'Ho' ---")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        class_channel = preds[:, 0]
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = last_conv_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    return (tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-6)).numpy()

def save_and_display_gradcam(img, heatmap, save_path, class_name, alpha=0.5):
    if isinstance(img, tf.Tensor): img = img.numpy()
    heatmap_resized = tf.image.resize(np.expand_dims(heatmap, -1), (img.shape[0], img.shape[1]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    jet = plt.cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_uint8.squeeze()]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)
    superimposed_img = tf.keras.utils.array_to_img(jet_heatmap * alpha + tf.keras.utils.img_to_array(img * 255))
    
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1); plt.imshow(img); plt.title(f"Ảnh Spectrogram Trung bình\nLớp: {class_name}"); plt.axis('off')
    plt.subplot(1, 2, 2); plt.imshow(superimposed_img); plt.title(f"Grad-CAM Trung bình - Vùng AI Chú ý"); plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

last_conv_layer_name = next((layer.name for layer in reversed(model.layers) if isinstance(layer, tf.keras.layers.Conv2D)), None)
if last_conv_layer_name:
    print(f"Sử dụng lớp '{last_conv_layer_name}' để tạo Grad-CAM.")
    heatmap_sum, spectrogram_sum, cough_sample_count = None, None, 0
    for spec_batch, label_batch in test_ds:
        cough_indices = tf.where(label_batch == 1).numpy().flatten()
        if len(cough_indices) > 0:
            cough_specs = tf.gather(spec_batch, cough_indices)
            if spectrogram_sum is None: spectrogram_sum = tf.reduce_sum(cough_specs, axis=0)
            else: spectrogram_sum += tf.reduce_sum(cough_specs, axis=0)
            for i in range(cough_specs.shape[0]):
                heatmap = make_gradcam_heatmap(np.expand_dims(cough_specs[i], 0), model, last_conv_layer_name)
                if heatmap_sum is None: heatmap_sum = heatmap
                else: heatmap_sum += heatmap
            cough_sample_count += len(cough_indices)
    if cough_sample_count > 0:
        avg_heatmap = heatmap_sum / cough_sample_count
        avg_spectrogram = spectrogram_sum / cough_sample_count
        save_path = os.path.join(FINAL_OUTPUT_PATH, f'{timestamp}_gradcam_avg_ho.png')
        save_and_display_gradcam(avg_spectrogram, avg_heatmap, save_path, "Ho")
    else:
        print("Không tìm thấy mẫu 'Ho' nào trong tập test để tạo Grad-CAM.")
else:
    print("Không tìm thấy lớp Conv2D để tạo Grad-CAM.")

print(f"\n--- Pipeline đã hoàn thành! Mọi kết quả đã được lưu tại {FINAL_OUTPUT_PATH} ---")