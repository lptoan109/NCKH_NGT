import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
import librosa
import numpy as np
import noisereduce as nr
import os

# --- 1. CẤU HÌNH (Đã chỉnh theo notebook huấn luyện) ---
SAMPLE_RATE = 16000
N_MELS = 128       # QUAN TRỌNG: Sửa từ 256 thành 128 theo log huấn luyện
N_FFT = 2048
HOP_LENGTH = 512
SILENCE_THRESHOLD_DB = 20
SEGMENT_LENGTH_S = 4
TARGET_SAMPLES = SEGMENT_LENGTH_S * SAMPLE_RATE 
CLASS_NAMES = ["asthma", "covid", "healthy", "tuberculosis"]
DEVICE = torch.device("cpu")

# ==========================================
# 2. ĐỊNH NGHĨA MÔ HÌNH: EffNetV2B0_CNN (Model 2)
# ==========================================
class EffNetV2B0_CNN(nn.Module):
    def __init__(self, num_classes=4, dropout=0.5):
        super(EffNetV2B0_CNN, self).__init__()
        
        # Chuyển 1 kênh (trắng đen) sang 3 kênh (màu) cho EfficientNet
        self.channel_converter = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        
        # Resize về 224x224 như lúc train
        self.resizer = transforms.Resize((224, 224), antialias=True)
        
        # Tải kiến trúc EfficientNet-B0 (Không tải weights pre-trained để tránh lỗi mạng)
        self.base_model = models.efficientnet_b0(weights=None)
        
        # Thay thế lớp classifier cuối cùng
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        # x shape: (batch, 1, 128, time)
        x = self.relu(self.bn(self.channel_converter(x))) 
        x = self.resizer(x) # Resize về (224, 224)
        output = self.base_model(x)
        return output

# ==========================================

# --- 3. Tải Model ---
print("Đang tải mô hình...")
model = EffNetV2B0_CNN(num_classes=len(CLASS_NAMES))
try:
    # Load trọng số từ file
    state_dict = torch.load("best_model.pth", map_location=DEVICE)
    
    # Xử lý trường hợp key có prefix 'module.' (nếu train nhiều GPU) hoặc sai lệch nhỏ
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
        
    model.load_state_dict(state_dict, strict=False) # strict=False để bỏ qua các lỗi nhỏ không quan trọng
    model.to(DEVICE)
    model.eval()
    print("-> Tải thành công Model 2 (EffNetV2B0_CNN)!")
except Exception as e:
    print(f"-> LỖI TẢI MODEL: {e}")

# --- 4. Hàm Tiền xử lý ---
def normalize_spectrogram(data):
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max == data_min: return np.zeros_like(data)
    return (data - data_min) / (data_max - data_min + 1e-8)

def process_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        y = librosa.util.normalize(y)
        y_denoised = nr.reduce_noise(y=y, sr=sr)
        y_trimmed, _ = librosa.effects.trim(y_denoised, top_db=SILENCE_THRESHOLD_DB)
        
        if len(y_trimmed) < 1: return None

        if len(y_trimmed) < TARGET_SAMPLES:
            y_padded = np.pad(y_trimmed, (0, TARGET_SAMPLES - len(y_trimmed)), 'constant')
        else:
            y_padded = y_trimmed[:TARGET_SAMPLES]
            
        # Tính Mel Spectrogram với N_MELS=128
        mel_spec = librosa.feature.melspectrogram(
            y=y_padded, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max, amin=1e-10)
        mel_spec_norm = normalize_spectrogram(mel_spec_db)

        # PyTorch shape: (1, 1, 128, time)
        tensor_input = torch.tensor(mel_spec_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        return tensor_input
        
    except Exception as e:
        print(f"Lỗi tiền xử lý: {e}")
        return None

# --- 5. Hàm dự đoán (Trả về đúng định dạng cho Web) ---
def predict(audio_file):
    if model is None: return {"Lỗi": "Mô hình chưa được tải"}
    if audio_file is None: return {"Lỗi": "Chưa có file âm thanh"}
    
    tensor_input = process_audio(audio_file)
    if tensor_input is None: return {"Lỗi": "File âm thanh không hợp lệ (quá ngắn hoặc im lặng)"}
    
    with torch.no_grad():
        tensor_input = tensor_input.to(DEVICE)
        outputs = model(tensor_input)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # Trả về dictionary {Tên bệnh: Xác suất}
    confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    return confidences

# --- 6. Giao diện ---
iface = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Tải lên file tiếng ho"),
    outputs=gr.Label(num_top_classes=4),
    title="NGT Cough AI - PyTorch (Model 2)",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()