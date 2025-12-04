import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
import librosa
import numpy as np
import os

# ==========================================
# 1. CẤU HÌNH (ĐỒNG BỘ VỚI NOTEBOOK TRAIN)
# ==========================================
SAMPLE_RATE = 22050        # Chuẩn: 22050Hz
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
SILENCE_THRESHOLD_DB = 20
SEGMENT_LENGTH_S = 5       # Chuẩn: 5 giây
TARGET_SAMPLES = int(SAMPLE_RATE * SEGMENT_LENGTH_S) # 110250 samples
TARGET_WIDTH = (TARGET_SAMPLES // HOP_LENGTH) + 1    # 216 frames
CLASS_NAMES = ["asthma", "covid", "healthy", "tuberculosis"]
DEVICE = torch.device("cpu")

print(f"Cấu hình Audio: SR={SAMPLE_RATE}, Duration={SEGMENT_LENGTH_S}s")

# ==========================================
# 2. ĐỊNH NGHĨA MÔ HÌNH
# ==========================================
class EffNetV2B0_CNN(nn.Module):
    def __init__(self, num_classes=4, dropout=0.5):
        super(EffNetV2B0_CNN, self).__init__()
        # Chuyển 1 kênh (Spectrogram) sang 3 kênh (RGB giả lập)
        self.channel_converter = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        # Resize về 224x224 (Input chuẩn EfficientNet)
        self.resizer = transforms.Resize((224, 224), antialias=True)
        # Tải kiến trúc EfficientNet-B0
        self.base_model = models.efficientnet_b0(weights=None)
        # Thay thế lớp classifier cuối cùng
        in_features = self.base_model.classifier[1].in_features
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(p=dropout, inplace=True),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        x = self.relu(self.bn(self.channel_converter(x))) 
        x = self.resizer(x)
        output = self.base_model(x)
        return output

# ==========================================
# 3. TẢI MODEL
# ==========================================
print("Đang tải mô hình...")
model = EffNetV2B0_CNN(num_classes=len(CLASS_NAMES))
try:
    state_dict = torch.load("best_model.pth", map_location=DEVICE)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    print("-> Tải thành công Model!")
except Exception as e:
    print(f"-> LỖI TẢI MODEL: {e}")

# ==========================================
# 4. HÀM TIỀN XỬ LÝ (GIỐNG NOTEBOOK)
# ==========================================
def process_audio(file_path):
    try:
        # 1. Load audio với SR chuẩn 22050
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # 2. Loại bỏ khoảng lặng
        y, _ = librosa.effects.trim(y, top_db=SILENCE_THRESHOLD_DB)
        
        # KIỂM TRA FILE RỖNG (Quan trọng)
        if len(y) <1:
            return None
        
        # 3. Pad/Crop audio về đúng độ dài 5s
        if len(y) > TARGET_SAMPLES:
            y = y[:TARGET_SAMPLES]
        elif len(y) < TARGET_SAMPLES:
            y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode='constant')
            
        # 4. Tạo Mel Spectrogram
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH
        )
        
        # 5. Chuyển sang dB
        S_db = librosa.power_to_db(S, ref=np.max)
        
        # 6. Chuẩn hóa Min-Max (về 0-1) - Khớp notebook
        min_val = S_db.min()
        max_val = S_db.max()
        S_db = (S_db - min_val) / (max_val - min_val + 1e-6)
        
        # 7. Đảm bảo đúng shape (128, 216)
        if S_db.shape[1] > TARGET_WIDTH:
            S_db = S_db[:, :TARGET_WIDTH]
        elif S_db.shape[1] < TARGET_WIDTH:
            S_db = np.pad(S_db, ((0,0), (0, TARGET_WIDTH - S_db.shape[1])), mode='constant')

        # Chuyển sang Tensor: (1, 1, 128, 216)
        tensor_input = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor_input
        
    except Exception as e:
        print(f"Lỗi tiền xử lý: {e}")
        return None

# ==========================================
# 5. HÀM DỰ ĐOÁN (RETURN DICTIONARY CHO API)
# ==========================================
def predict(audio_file):
    # 1. Kiểm tra model
    if model is None: 
        return {"Lỗi": "Server"}
    
    # 2. Kiểm tra file input
    if audio_file is None: 
        return {"Lỗi": "Chưa có file âm thanh"}
    
    # 3. Tiền xử lý
    tensor_input = process_audio(audio_file)
    if tensor_input is None: 
        return {"Lỗi": "File âm thanh không hợp lệ (quá ngắn hoặc im lặng)"}
    
    # 4. Dự đoán
    with torch.no_grad():
        tensor_input = tensor_input.to(DEVICE)
        outputs = model(tensor_input)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    # 5. Trả về kết quả dạng Dictionary {Label: Probability}
    # Đây là định dạng chuẩn để web app có thể đọc được JSON
    confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    
    return confidences

# ==========================================
# 6. GIAO DIỆN
# ==========================================
iface = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Tải lên file tiếng ho"),
    outputs=gr.Label(num_top_classes=4),
    title="NGT Cough AI - PyTorch (Final)",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()