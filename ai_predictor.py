import os
import numpy as np
import noisereduce as nr
import soundfile as sf
import tflite_runtime.interpreter as tflite
from tensorflow.keras.applications.efficientnet import preprocess_input
import audio_utils_light as audio # Sử dụng file helper đã tạo

# Các hằng số cho việc xử lý âm thanh
SAMPLE_RATE = 16000
MIN_DURATION_S = 1.0
SILENCE_THRESHOLD_DB = 40

# Các hằng số cho Mel Spectrogram
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

class CoughPredictor:
    def __init__(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        # --- THÊM VÀO: Khởi tạo nhãn lớp ---
        # THAY ĐỔI THỨ TỰ NÀY ĐỂ KHỚP VỚI LÚC HUẤN LUYỆN
        self.labels = ['covid', 'asthma', 'healthy', 'tuberculosis'] 
    
    def _softmax(self, x):
        """Tính toán softmax cho một mảng numpy."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def _process_audio(self, file_path):
        """
        Hàm private để xử lý một file âm thanh và trả về các tensor spectrogram.
        (Phần này đã đúng logic, giữ nguyên)
        """
        try:
            y, sr = sf.read(file_path, dtype='float32')

            if sr != SAMPLE_RATE:
                print(f"Cảnh báo: Tần số lấy mẫu của file là {sr}Hz, không phải {SAMPLE_RATE}Hz. Kết quả có thể không chính xác.")
            y = audio.to_mono(y)

            if len(y) / sr < MIN_DURATION_S:
                return None

            y = audio.normalize(y)
            y_denoised = nr.reduce_noise(y=y, sr=sr)
            y_trimmed = audio.trim(y_denoised, top_db=SILENCE_THRESHOLD_DB)

            if len(y_trimmed) < N_FFT:
                return None

            mel_spec = audio.melspectrogram(
                y=y_trimmed, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
            )
            db_spec = audio.power_to_db(mel_spec, top_db=80)
            
            spec_rgb = np.stack([db_spec, db_spec, db_spec], axis=-1)
            spec_expanded = np.expand_dims(spec_rgb, axis=0)
            preprocessed_spec = preprocess_input(spec_expanded)

            return preprocessed_spec

        except Exception as e:
            print(f"Lỗi khi xử lý file {os.path.basename(file_path)}: {e}")
            return None

    def predict(self, audio_file_path):
        """
        Hàm dự đoán đã được sửa lại cho đúng logic.
        """
        # 1. Xử lý audio để lấy tensor đầu vào
        input_tensor = self._process_audio(audio_file_path)
        
        if input_tensor is None:
            return {"error": "Không thể xử lý file âm thanh (quá ngắn hoặc lỗi)."}

        # 2. Đặt tensor đầu vào cho mô hình TFLite
        self.interpreter.set_tensor(self.input_details[0]['index'], input_tensor.astype(np.float32))
        
        # 3. Chạy dự đoán
        self.interpreter.invoke()
        
        # 4. Lấy kết quả đầu ra (dạng logits)
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predictions_logits = output_data[0] # Lấy mảng 1D từ output
        
        # 5. Áp dụng softmax để chuyển thành xác suất
        prediction_probs = self._softmax(predictions_logits)
        
        # 6. Xác định lớp và độ tin cậy
        predicted_class_index = np.argmax(prediction_probs)
        predicted_class_name = self.labels[predicted_class_index]
        confidence = float(prediction_probs[predicted_class_index])
        
        # 7. Định dạng kết quả trả về
        display_names = {"healthy": "Khỏe mạnh", "asthma": "Hen suyễn", "covid": "COVID-19", "tuberculosis": "Lao"}

        return {
            "predicted_class": display_names.get(predicted_class_name, predicted_class_name),
            "confidence": f"{confidence:.2%}",
            "details": {display_names.get(label, label): f"{prob:.2%}" for label, prob in zip(self.labels, prediction_probs)}
        }