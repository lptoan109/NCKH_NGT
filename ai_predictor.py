import os
import numpy as np
import librosa
import noisereduce as nr
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- CÁC HẰNG SỐ TIỀN XỬ LÝ (PHẢI GIỐNG HỆT KHI HUẤN LUYỆN) ---
SAMPLE_RATE = 16000
MIN_DURATION_S = 2.0
SILENCE_THRESHOLD_DB = 20
SEGMENT_LENGTH_S = 4.0
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128
INPUT_SHAPE = (240, 240, 3) # Kích thước input cho EfficientNetB1

# --- LỚP ĐIỀU KHIỂN TOÀN BỘ LOGIC AI ---
class CoughPredictor:
    def __init__(self, model_path):
        """
        Khởi tạo và tải mô hình AI vào bộ nhớ.
        Args:
            model_path (str): Đường dẫn đến file .keras của mô hình học sinh.
        """
        print(f"--- Đang tải mô hình từ: {model_path} ---")
        self.model = tf.keras.models.load_model(model_path)
        self.labels = ['healthy', 'asthma', 'covid', 'tuberculosis'] # Phải đúng thứ tự khi huấn luyện
        print("--- Mô hình đã sẵn sàng! ---")

    def _process_audio(self, file_path):
        """
        Hàm private để xử lý một file âm thanh và trả về các tensor spectrogram.
        Đây là phiên bản đã được tối ưu cho việc dự đoán (inference).
        """
        try:
            y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
            if len(y) / sr < MIN_DURATION_S:
                return None

            y = librosa.util.normalize(y)
            y_denoised = nr.reduce_noise(y=y, sr=sr)
            y_trimmed, _ = librosa.effects.trim(y_denoised, top_db=SILENCE_THRESHOLD_DB)

            if len(y_trimmed) < 1:
                return None

            segment_samples = int(SEGMENT_LENGTH_S * sr)
            spectrograms = []

            for i in range(0, len(y_trimmed), segment_samples):
                segment = y_trimmed[i:i + segment_samples]
                if len(segment) < segment_samples:
                    padding = segment_samples - len(segment)
                    segment = np.pad(segment, (0, padding), 'constant')

                mels = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS)
                mels_db = librosa.power_to_db(mels, ref=np.max)
                spectrograms.append(mels_db)

            if not spectrograms:
                return None

            spectrograms_np = np.array(spectrograms).astype(np.float32)
            return spectrograms_np

        except Exception as e:
            print(f"Lỗi xử lý file âm thanh: {e}")
            return None
            
    def _prepare_tensors_for_model(self, specs_np):
        """
        Hàm private để chuẩn hóa và định hình lại spectrogram cho đúng input của mô hình.
        """
        # Thêm chiều kênh và lặp lại 3 lần
        specs_3d = np.stack([specs_np]*3, axis=-1)

        # Chuẩn hóa dải giá trị về [0, 255]
        min_val = specs_3d.min()
        max_val = specs_3d.max()
        scaled_01 = (specs_3d - min_val) / (max_val - min_val + 1e-7)
        scaled_255 = scaled_01 * 255.0

        # Resize về kích thước input
        input_tensors = tf.image.resize(scaled_255, [INPUT_SHAPE[0], INPUT_SHAPE[1]])
        
        # Áp dụng hàm tiền xử lý của EfficientNet
        final_tensors = preprocess_input(input_tensors)
        
        return final_tensors


    def predict(self, audio_file_path):
        spectrograms = self._process_audio(audio_file_path)
        if spectrograms is None:
            return {"error": "Không thể xử lý file âm thanh."}

        input_tensors = self._prepare_tensors_for_model(spectrograms)
        
        predictions_logits = self.model.predict(input_tensors)
        predictions_probs = tf.nn.softmax(predictions_logits).numpy()

        avg_prediction_probs = np.mean(predictions_probs, axis=0)
        
        predicted_class_index = np.argmax(avg_prediction_probs)
        predicted_class_name = self.labels[predicted_class_index]
        confidence = float(avg_prediction_probs[predicted_class_index])
        
        # Đổi tên lớp để hiển thị thân thiện hơn
        display_names = {
            "healthy": "Khỏe mạnh",
            "asthma": "Hen suyễn",
            "covid": "COVID-19",
            "tuberculosis": "Lao"
        }
        
        return {
            "predicted_class": display_names.get(predicted_class_name, predicted_class_name),
            "confidence": f"{confidence:.2%}"
        }