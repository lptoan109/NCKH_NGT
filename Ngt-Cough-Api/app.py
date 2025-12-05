import gradio as gr
import torch
import torch.nn as nn
from torchvision import models, transforms
import librosa
import numpy as np
import noisereduce as nr  # <--- Đã thêm thư viện lọc nhiễu
import os
import warnings

# --- TẮT CẢNH BÁO RÁC ---
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message="PySoundFile failed")
# -------------------------

# ==========================================
# 1. CẤU HÌNH
# ==========================================
SAMPLE_RATE = 22050        # Chuẩn: 22050Hz
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
SILENCE_THRESHOLD_DB = 20
SEGMENT_LENGTH_S = 5       # Chuẩn: 5 giây
TARGET_SAMPLES = int(SAMPLE_RATE * SEGMENT_LENGTH_S) 
TARGET_WIDTH = (TARGET_SAMPLES // HOP_LENGTH) + 1    
CLASS_NAMES = ["asthma", "covid", "healthy", "tuberculosis"]
DEVICE = torch.device("cpu")

print(f"Cấu hình Audio: SR={SAMPLE_RATE}, Duration={SEGMENT_LENGTH_S}s")

# ==========================================
# 2. ĐỊNH NGHĨA MÔ HÌNH
# ==========================================
class EffNetV2B0_CNN(nn.Module):
    def __init__(self, num_classes=4, dropout=0.5):
        super(EffNetV2B0_CNN, self).__init__()
        self.channel_converter = nn.Conv2d(1, 3, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self.resizer = transforms.Resize((224, 224), antialias=True)
        self.base_model = models.efficientnet_b0(weights=None)
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
# 4. HÀM TIỀN XỬ LÝ
# ==========================================
def process_audio(file_path):
    try:
        # 1. Load audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # --- BƯỚC MỚI: LỌC NHIỄU (DENOISE) ---
        try:
            # stationarity=True giúp lọc nhiễu nền ổn định (tiếng quạt, gió...)
            # prop_decrease=1.0: Mức độ giảm nhiễu tối đa
            y = nr.reduce_noise(y=y, sr=sr, stationary=True, prop_decrease=1.0)
        except Exception as e:
            print(f"Cảnh báo: Không thể lọc nhiễu ({e}), tiếp tục xử lý gốc.")
        # -------------------------------------

        # 2. Loại bỏ khoảng lặng
        y, _ = librosa.effects.trim(y, top_db=SILENCE_THRESHOLD_DB)
        
        # Kiểm tra file rỗng
        if len(y) == 0:
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
        
        # 6. Chuẩn hóa Min-Max (0-1)
        min_val = S_db.min()
        max_val = S_db.max()
        S_db = (S_db - min_val) / (max_val - min_val + 1e-6)
        
        # 7. Đảm bảo đúng shape (128, 216)
        if S_db.shape[1] > TARGET_WIDTH:
            S_db = S_db[:, :TARGET_WIDTH]
        elif S_db.shape[1] < TARGET_WIDTH:
            S_db = np.pad(S_db, ((0,0), (0, TARGET_WIDTH - S_db.shape[1])), mode='constant')

        # Chuyển sang Tensor
        tensor_input = torch.tensor(S_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return tensor_input
        
    except Exception as e:
        print(f"Lỗi tiền xử lý: {e}")
        return None

# ==========================================
# 5. HÀM DỰ ĐOÁN
# ==========================================
def predict(audio_file):
    if model is None: 
        return {"Lỗi": "Mô hình chưa được tải (thiếu file .pth)"}
    
    if audio_file is None: 
        return {"Lỗi": "Chưa có file âm thanh"}
    
    tensor_input = process_audio(audio_file)
    if tensor_input is None: 
        return {"Lỗi": "File âm thanh không hợp lệ (quá ngắn hoặc im lặng)"}
    
    with torch.no_grad():
        tensor_input = tensor_input.to(DEVICE)
        outputs = model(tensor_input)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    
    confidences = {CLASS_NAMES[i]: float(probabilities[i]) for i in range(len(CLASS_NAMES))}
    return confidences

# ==========================================
# 6. GIAO DIỆN
# ==========================================
iface = gr.Interface(
    fn=predict,
    inputs=gr.Audio(type="filepath", label="Tải lên file tiếng ho"),
    outputs=gr.Label(num_top_classes=4),
    title="NGT Cough AI - PyTorch (Có lọc nhiễu)",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()