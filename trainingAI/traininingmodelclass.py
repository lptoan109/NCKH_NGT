# ======================================================================================
# BLOCK 1: CÀI ĐẶT, IMPORT VÀ CẤU HÌNH NÂNG CAO (Version Final - Đã sửa lỗi font)
# ======================================================================================
print("BLOCK 1: CÀI ĐẶT, IMPORT VÀ CẤU HÌNH NÂNG CAO...")

# Cài đặt các thư viện cần thiết
!pip install timm torch torchaudio scikit-learn pandas matplotlib seaborn librosa pydub pytz reportlab grad-cam transformers -q

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import timm
from torchvision import transforms
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from pydub import AudioSegment
from pydub.silence import split_on_silence
from torch.optim import AdamW
from transformers import get_scheduler
from google.colab import drive
import datetime
import pytz
from tqdm.auto import tqdm
import requests # Dùng thư viện này để tải font
from itertools import cycle
from PIL import Image

# Thư viện cho PDF Report và Grad-CAM
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as ReportlabImage, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- THAY ĐỔI QUAN TRỌNG: Tải và đăng ký font một cách an toàn hơn ---
VIETNAMESE_FONT_PATH = "Roboto-Regular.ttf"
FONT_URL = "https://github.com/google/fonts/raw/main/ofl/roboto/Roboto-Regular.ttf"

print("Dang tai font tieng Viet cho bao cao PDF...")
try:
    response = requests.get(FONT_URL)
    if response.status_code == 200:
        with open(VIETNAMESE_FONT_PATH, "wb") as f:
            f.write(response.content)
        pdfmetrics.registerFont(TTFont('Roboto', VIETNAMESE_FONT_PATH))
        print("Tai va dang ky font 'Roboto' thanh cong.")
    else:
        print(f"Loi khi tai font: HTTP Status Code {response.status_code}")
except Exception as e:
    print(f"Da xay ra loi khi tai hoac dang ky font: {e}")


# --- Cấu hình toàn bộ pipeline ---
class CONFIG:
    # --- CÔNG TẮC KHÔI PHỤC ---
    RESUME_TRAINING = False
    RESUME_SESSION_FOLDER_NAME = ""

    # --- Cấu hình Mô hình & Huấn luyện ---
    MODEL_ID = "efficientnet_b3"
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 16

    # --- Cấu hình Chống Overfitting ---
    EARLY_STOPPING_PATIENCE = 10

    # --- Cấu hình Dữ liệu & Tiền xử lý ---
    # !!! THAY ĐỔI ĐƯỜNG DẪN NÀY !!!
    # Đường dẫn đến thư mục gốc trên Google Drive chứa các thư mục con của dữ liệu âm thanh
    # Ví dụ: "/content/drive/MyDrive/NCKH/BoDuLieuAmThanh"
    DATA_PATH = "/content/drive/MyDrive/Tai_Lieu_NCKH/dataset_unhealthy"
    IMAGE_SIZE = 300
    SAMPLE_RATE = 16000
    MAX_LENGTH_SECS = 5
    N_MELS = 224
    N_FFT = 2048
    HOP_LENGTH = 512
    SILENCE_THRESH = -40
    MIN_SILENCE_LEN = 300

    # --- Cấu hình Google Drive & Output ---
    DRIVE_MOUNT_PATH = "/content/drive"
    # !!! THAY ĐỔI ID NÀY !!!
    # ID của thư mục trên Google Drive để lưu tất cả kết quả (checkpoint, báo cáo...)
    DRIVE_OUTPUT_FOLDER_ID = "1T2QPRUEIGWjmaHfgBEswyupEad_7WqWm"
    CHECKPOINT_SUBFOLDER = "checkpoints"

    # --- Cấu hình Timezone ---
    TIMEZONE = "Asia/Ho_Chi_Minh"

# --- Setup Môi trường ---
drive.mount(CONFIG.DRIVE_MOUNT_PATH)
BASE_DRIVE_OUTPUT_PATH = os.path.join(CONFIG.DRIVE_MOUNT_PATH, "MyDrive", CONFIG.DRIVE_OUTPUT_FOLDER_ID)
os.makedirs(BASE_DRIVE_OUTPUT_PATH, exist_ok=True)

tz = pytz.timezone(CONFIG.TIMEZONE)
def get_vn_time_str():
    return datetime.datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")

if CONFIG.RESUME_TRAINING and CONFIG.RESUME_SESSION_FOLDER_NAME:
    SESSION_FOLDER = os.path.join(BASE_DRIVE_OUTPUT_PATH, CONFIG.RESUME_SESSION_FOLDER_NAME)
    print(f"Che do RESUME: Se tiep tuc tu session tai: {SESSION_FOLDER}")
else:
    SESSION_FOLDER = os.path.join(BASE_DRIVE_OUTPUT_PATH, f"session_{get_vn_time_str()}")
    os.makedirs(SESSION_FOLDER, exist_ok=True)
    print(f"Che do MOI: Output se duoc luu tai: {SESSION_FOLDER}")

CHECKPOINT_FOLDER = os.path.join(SESSION_FOLDER, CONFIG.CHECKPOINT_SUBFOLDER)
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
print("Hoan tat Block 1.")


# ======================================================================================
# BLOCK 2: CÁC HÀM TIỀN XỬ LÝ ÂM THANH VÀ SPECTROGRAM
# ======================================================================================
print("\nBLOCK 2: KHỞI TẠO CÁC HÀM TIỀN XỬ LÝ...")
def preprocess_audio(file_path):
    try:
        sound = AudioSegment.from_file(file_path, format="wav")
        sound = sound.set_frame_rate(CONFIG.SAMPLE_RATE).set_channels(1)
        chunks = split_on_silence(sound, min_silence_len=CONFIG.MIN_SILENCE_LEN, silence_thresh=CONFIG.SILENCE_THRESH)
        processed_sound = sum(chunks, AudioSegment.empty()) if chunks else sound
        samples = np.array(processed_sound.get_array_of_samples()).astype(np.float32)
        samples /= np.iinfo(processed_sound.sample_width * 8).max
        max_samples = CONFIG.SAMPLE_RATE * CONFIG.MAX_LENGTH_SECS
        if len(samples) > max_samples: samples = samples[:max_samples]
        else: samples = np.pad(samples, (0, max_samples - len(samples)), 'constant')
        return samples
    except Exception as e:
        print(f"Lỗi xử lý file {file_path}: {e}"); return None

def waveform_to_spectrogram(waveform):
    mel_spectrogram = librosa.feature.melspectrogram(
        y=waveform, sr=CONFIG.SAMPLE_RATE, n_fft=CONFIG.N_FFT,
        hop_length=CONFIG.HOP_LENGTH, n_mels=CONFIG.N_MELS
    )
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return log_mel_spectrogram
print("Hoàn tất Block 2.")


# ======================================================================================
# BLOCK 3: DATASET VÀ DATALOADER (Đã sửa lỗi kiểu dữ liệu của nhãn)
# ======================================================================================
print("\nBLOCK 3: KHỞI TẠO DATASET VÀ DATALOADER...")

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

# Định nghĩa các phép biến đổi cho ảnh spectrogram
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
}

# Định nghĩa lớp Dataset
class CoughDataset(Dataset):
    def __init__(self, file_paths, labels, transform):
        self.file_paths, self.labels, self.transform = file_paths, labels, transform
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        waveform = preprocess_audio(self.file_paths[idx])
        if waveform is None:
            # --- THAY ĐỔI QUAN TRỌNG: Chuyển số nguyên -1 thành Tensor ---
            return {"image": torch.zeros(3, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE), "labels": torch.tensor(-1, dtype=torch.long)}

        spectrogram = waveform_to_spectrogram(waveform)
        spec_normalized = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        spec_img = Image.fromarray((spec_normalized * 255).astype(np.uint8)).convert("RGB")
        image_tensor = self.transform(spec_img)
        return {"image": image_tensor, "labels": torch.tensor(self.labels[idx], dtype=torch.long)}

print("Thực hiện chia dữ liệu theo nhóm (Group Split)...")

all_data = []
class_names = sorted([d for d in os.listdir(CONFIG.DATA_PATH) if os.path.isdir(os.path.join(CONFIG.DATA_PATH, d))])
class_to_idx = {name: i for i, name in enumerate(class_names)}
idx_to_class = {i: name for i, name in enumerate(class_names)}

for class_name in class_names:
    class_dir = os.path.join(CONFIG.DATA_PATH, class_name)
    for file_name in os.listdir(class_dir):
        if file_name.lower().endswith(".wav"):
            file_path = os.path.join(class_dir, file_name)
            participant_id = ""
            if class_name in ['healthy', 'covid', 'asthma']:
                participant_id = file_name.split('_')[0]
            elif class_name == 'tuberculosis':
                if '_' in file_name: participant_id = file_name.rsplit('_', 1)[0]
                else: participant_id = file_name
            all_data.append({"file_path": file_path, "label": class_to_idx[class_name], "participant_id": participant_id})

df = pd.DataFrame(all_data)
print(f"Đã quét {len(df)} files từ {df['participant_id'].nunique()} cá nhân duy nhất.")

splitter = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=42)
train_idx, temp_idx = next(splitter.split(df, groups=df['participant_id']))
train_df, temp_df = df.iloc[train_idx], df.iloc[temp_idx]

splitter_val_test = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=42)
val_idx, test_idx = next(splitter_val_test.split(temp_df, groups=temp_df['participant_id']))
val_df, test_df = temp_df.iloc[val_idx], temp_df.iloc[test_idx]

train_files, train_labels = train_df['file_path'].tolist(), train_df['label'].tolist()
val_files, val_labels = val_df['file_path'].tolist(), val_df['label'].tolist()
test_files, test_labels = test_df['file_path'].tolist(), test_df['label'].tolist()

print(f"\nChia dữ liệu hoàn tất:")
print(f" - Tập Train: {len(train_files)} files từ {len(train_df['participant_id'].unique())} người.")
print(f" - Tập Validation: {len(val_files)} files từ {len(val_df['participant_id'].unique())} người.")
print(f" - Tập Test: {len(test_files)} files từ {len(test_df['participant_id'].unique())} người.")

train_dataset = CoughDataset(train_files, train_labels, data_transforms['train'])
val_dataset = CoughDataset(val_files, val_labels, data_transforms['val'])
test_dataset = CoughDataset(test_files, test_labels, data_transforms['val'])

train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, num_workers=2)

print("\nHoàn tất Block 3.")


# ======================================================================================
# BLOCK 4: MÔ HÌNH VÀ CÁC TIỆN ÍCH NÂNG CAO (Đã sửa lỗi np.Inf)
# ======================================================================================
print("\nBLOCK 4: KHỞI TẠO MÔ HÌNH VÀ CÁC TIỆN ÍCH...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

def load_model(num_labels, pretrained=True):
    model = timm.create_model(CONFIG.MODEL_ID, pretrained=pretrained, num_classes=num_labels)
    return model.to(device)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth.tar'):
        self.patience, self.verbose, self.delta, self.path = patience, verbose, delta, path
        self.counter, self.best_score, self.early_stop = 0, None, False
        # --- THAY ĐỔI QUAN TRỌNG: Sửa np.Inf thành np.inf ---
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model_state):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            if self.verbose: print(f'Validation loss giảm ({self.val_loss_min:.6f} --> {val_loss:.6f}). Đang lưu model...')
            torch.save(model_state, self.path)
            self.val_loss_min, self.best_score, self.counter = val_loss, score, 0
        else:
            self.counter += 1
            if self.verbose: print(f'EarlyStopping counter: {self.counter} / {self.patience}')
            if self.counter >= self.patience: self.early_stop = True

def load_checkpoint(model, optimizer, scheduler, folder):
    filepath = os.path.join(folder, "last_checkpoint.pth.tar")
    if os.path.exists(filepath):
        print(f"Tải checkpoint từ {filepath}")
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'] + 1, checkpoint['history']
    return 0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

def save_checkpoint(state, folder):
    filepath = os.path.join(folder, "last_checkpoint.pth.tar")
    torch.save(state, filepath)
print("Hoàn tất Block 4.")


# ======================================================================================
# BLOCK 5: VÒNG LẶP HUẤN LUYỆN
# ======================================================================================
print("\nBLOCK 5: BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN...")
model = load_model(len(class_names))
optimizer = AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
num_training_steps = CONFIG.EPOCHS * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1*num_training_steps), num_training_steps=num_training_steps)
early_stopper = EarlyStopping(patience=CONFIG.EARLY_STOPPING_PATIENCE, verbose=True, path=os.path.join(CHECKPOINT_FOLDER, "best_model.pth.tar"))
start_epoch, history = load_checkpoint(model, optimizer, lr_scheduler, CHECKPOINT_FOLDER) if CONFIG.RESUME_TRAINING else (0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []})

for epoch in range(start_epoch, CONFIG.EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{CONFIG.EPOCHS} ---")
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        if -1 in batch['labels']: continue
        images, labels = batch['image'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step(); lr_scheduler.step()
        predictions = torch.argmax(logits, dim=-1)
        total_loss += loss.item() * images.size(0)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        progress_bar.set_postfix(loss=total_loss/total_samples, acc=total_correct/total_samples)
    train_loss, train_acc = total_loss / total_samples, total_correct / total_samples
    history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)

    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            if -1 in batch['labels']: continue
            images, labels = batch['image'].to(device), batch['labels'].to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            total_correct += (torch.argmax(logits, dim=-1) == labels).sum().item()
            total_samples += labels.size(0)
    val_loss, val_acc = total_loss / total_samples, total_correct / total_samples
    history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
    print(f"Epoch {epoch + 1} Summary: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

    save_checkpoint({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': lr_scheduler.state_dict(), 'history': history}, CHECKPOINT_FOLDER)
    early_stopper(val_loss, model.state_dict())
    if early_stopper.early_stop: print("Early stopping triggered!"); break
print("\nHoàn tất quá trình huấn luyện.")


# ======================================================================================
# BLOCK 6: PHÂN TÍCH TOÀN DIỆN VÀ BÁO CÁO KẾT QUẢ (Đã cập nhật Grad-CAM trung bình)
# ======================================================================================
print("\nBLOCK 6: BẮT ĐẦU ĐÁNH GIÁ, PHÂN TÍCH SÂU VÀ TẠO BÁO CÁO...")

# --- 0. Tải lại mô hình tốt nhất để đánh giá ---
best_model_path = os.path.join(CHECKPOINT_FOLDER, "best_model.pth.tar")
if os.path.exists(best_model_path):
    print(f"Tải lại trọng số tốt nhất từ: {best_model_path}")
    model = load_model(len(class_names), pretrained=False)
    model.load_state_dict(torch.load(best_model_path))
else:
    print("Cảnh báo: Không tìm thấy best_model.pth.tar. Sử dụng mô hình cuối cùng để đánh giá.")

current_time_str = get_vn_time_str()
model.eval()

# --- 1. Thu thập tất cả kết quả, xác suất và embedding từ tập Test ---
print("Bắt đầu thu thập kết quả, xác suất và embeddings từ tập Test...")
all_preds, all_true, all_probs, all_embeddings = [], [], [], []
test_filepaths_in_order = [path for path in test_dataset.file_paths]
with torch.no_grad():
    for batch in tqdm(test_loader, "Collecting Test Set Results"):
        if -1 in batch['labels']: continue
        images, labels = batch['image'].to(device), batch['labels'].to(device)
        embeddings = model.forward_features(images)
        embeddings = model.global_pool(embeddings).flatten(1)
        logits = model.classifier(embeddings)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)
        all_probs.extend(probs.cpu().numpy()); all_preds.extend(predictions.cpu().numpy()); all_true.extend(labels.cpu().numpy()); all_embeddings.extend(embeddings.cpu().numpy())

all_true, all_preds, all_probs, all_embeddings = np.array(all_true), np.array(all_preds), np.array(all_probs), np.array(all_embeddings)

# --- 2. Tạo và lưu các biểu đồ hiệu suất ---
# (Không thay đổi)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10)); ax1.plot(history['train_loss'], label='Train Loss'); ax1.plot(history['val_loss'], label='Validation Loss'); ax1.set_title('Training & Validation Loss'); ax1.legend(); ax1.grid(True); ax2.plot(history['train_acc'], label='Train Accuracy'); ax2.plot(history['val_acc'], label='Validation Accuracy'); ax2.set_title('Training & Validation Accuracy'); ax2.legend(); ax2.grid(True)
metrics_plot_path = os.path.join(SESSION_FOLDER, f"metrics_plot_{current_time_str}.png"); plt.tight_layout(); plt.savefig(metrics_plot_path); plt.close(fig)
cm = confusion_matrix(all_true, all_preds); plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names); plt.title('Confusion Matrix')
cm_plot_path = os.path.join(SESSION_FOLDER, f"confusion_matrix_{current_time_str}.png"); plt.tight_layout(); plt.savefig(cm_plot_path); plt.close()
y_true_binarized = label_binarize(all_true, classes=range(len(class_names))); fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8)); line_colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
for i, color in zip(range(len(class_names)), line_colors):
    fpr, tpr, _ = roc_curve(y_true_binarized[:, i], all_probs[:, i]); roc_auc = auc(fpr, tpr); ax1.plot(fpr, tpr, color=color, lw=2, label=f'ROC {class_names[i]} (AUC = {roc_auc:0.2f})')
    precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], all_probs[:, i]); ax2.plot(recall, precision, color=color, lw=2, label=f'PR {class_names[i]}')
ax1.plot([0, 1], [0, 1], 'k--', lw=2); ax1.set_title('Multi-class ROC Curve'); ax1.legend(loc="lower right"); ax1.grid(True); ax2.set_title('Multi-class Precision-Recall Curve'); ax2.legend(loc="best"); ax2.grid(True)
roc_pr_plot_path = os.path.join(SESSION_FOLDER, f"roc_pr_plot_{current_time_str}.png"); plt.savefig(roc_pr_plot_path); plt.close(fig)
print(f"Đã lưu các biểu đồ hiệu suất.")

# --- 3. Trực quan hóa Embedding (t-SNE) ---
# (Không thay đổi)
print("\nBắt đầu tạo biểu đồ t-SNE..."); tsne = TSNE(n_components=2, verbose=0, perplexity=min(30, len(all_embeddings)-1), n_iter=300, random_state=42); tsne_results = tsne.fit_transform(all_embeddings); df_tsne = pd.DataFrame(tsne_results, columns=['tsne-1', 'tsne-2']); df_tsne['label'] = [idx_to_class[i] for i in all_true]
plt.figure(figsize=(10, 10)); sns.scatterplot(x="tsne-1", y="tsne-2", hue="label", palette=sns.color_palette("hsv", len(class_names)), data=df_tsne, legend="full", alpha=0.8); plt.title("t-SNE Visualization of Cough Embeddings"); plt.grid(True)
tsne_plot_path = os.path.join(SESSION_FOLDER, f"tsne_plot_{current_time_str}.png"); plt.savefig(tsne_plot_path); plt.close()
print(f"Đã lưu biểu đồ t-SNE.")

# --- 4. Tái cấu trúc hàm và Phân tích Lỗi Sai (Grad-CAM) ---

# Tách logic tạo Grad-CAM ra một hàm riêng để tái sử dụng
def get_grad_cam_array(model, file_path, target_class_idx):
    try:
        target_layers = [model.conv_head]
        cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
        waveform = preprocess_audio(file_path)
        if waveform is None: return None, None
        
        spectrogram = waveform_to_spectrogram(waveform)
        spec_normalized = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())
        spec_rgb = np.stack([spec_normalized]*3, axis=-1)
        
        input_tensor = data_transforms['val'](Image.fromarray((spec_rgb * 255).astype(np.uint8))).unsqueeze(0)
        
        targets = [ClassifierOutputTarget(target_class_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        
        return grayscale_cam, spec_rgb
    except Exception as e:
        print(f"Lỗi khi tạo Grad-CAM cho {file_path}: {e}")
        return None, None

# Hàm cũ được giữ lại để visualize lỗi sai
def visualize_single_grad_cam(model, file_path, output_path, pred_label_idx):
    grayscale_cam, spec_rgb = get_grad_cam_array(model, file_path, pred_label_idx)
    if grayscale_cam is not None:
        visualization = show_cam_on_image(spec_rgb, grayscale_cam, use_rgb=True)
        Image.fromarray(visualization).save(output_path)

print("\nBắt đầu phân tích các mẫu bị phân loại sai...")
misclassified_indices = np.where(all_preds != all_true)[0]
error_analysis, error_xai_paths = [], []
for idx in misclassified_indices: error_analysis.append((all_probs[idx][all_preds[idx]], all_true[idx], all_preds[idx], test_filepaths_in_order[idx]))
error_analysis.sort(key=lambda x: x[0], reverse=True)
print(f"\n--- Top 5 Lỗi sai Tự tin nhất ---")
for i, (confidence, true_label_idx, pred_label_idx, file_path) in enumerate(error_analysis[:5]):
    error_xai_path = os.path.join(SESSION_FOLDER, f"error_analysis_sample_{i+1}_{current_time_str}.png")
    visualize_single_grad_cam(model, file_path, error_xai_path, pred_label_idx)
    error_info = (f"Lỗi #{i+1}: '{os.path.basename(file_path)}' | Thật: '{idx_to_class[true_label_idx]}' | Sai: '{idx_to_class[pred_label_idx]}' (Conf: {confidence:.2f})", error_xai_path)
    error_xai_paths.append(error_info); print(error_info[0])

# --- 4.5.  Phân tích Grad-CAM trung bình cho các dự đoán ĐÚNG ---
print("\nBắt đầu tạo Grad-CAM trung bình cho các dự đoán đúng...")
avg_cam_paths = []
correct_indices = np.where(all_preds == all_true)[0]
df_correct = pd.DataFrame({
    'filepath': np.array(test_filepaths_in_order)[correct_indices],
    'true_label': all_true[correct_indices],
    'confidence': np.max(all_probs[correct_indices], axis=1)
})

for class_idx, class_name in idx_to_class.items():
    print(f" - Đang xử lý lớp: {class_name}")
    
    # Lọc ra top 10 file dự đoán đúng và tự tin nhất cho lớp hiện tại
    top_10_df = df_correct[df_correct['true_label'] == class_idx].nlargest(10, 'confidence')
    
    if top_10_df.empty:
        print(f"   -> Không có dự đoán đúng nào cho lớp {class_name} trong tập test.")
        continue
        
    cam_arrays = []
    # Lấy spectrogram nền từ file tự tin nhất
    background_spec = None
    
    for i, row in top_10_df.iterrows():
        grayscale_cam, spec_rgb = get_grad_cam_array(model, row['filepath'], class_idx)
        if grayscale_cam is not None:
            cam_arrays.append(grayscale_cam)
            if background_spec is None: # Lấy ảnh nền của file đầu tiên (tự tin nhất)
                background_spec = spec_rgb
    
    if not cam_arrays:
        print(f"   -> Không thể tạo Grad-CAM cho lớp {class_name}.")
        continue

    # Tính trung bình các heatmap
    avg_cam = np.mean(cam_arrays, axis=0)
    
    # Tạo ảnh visualization
    avg_visualization = show_cam_on_image(background_spec, avg_cam, use_rgb=True)
    
    # Lưu ảnh
    avg_cam_path = os.path.join(SESSION_FOLDER, f"average_gradcam_{class_name}_{current_time_str}.png")
    Image.fromarray(avg_visualization).save(avg_cam_path)
    
    avg_cam_info = (f"Grad-CAM Trung bình cho lớp '{class_name}' (dựa trên {len(cam_arrays)} mẫu)", avg_cam_path)
    avg_cam_paths.append(avg_cam_info)
    print(f"   -> Đã lưu Grad-CAM trung bình tại: {avg_cam_path}")


# --- 5. Tạo báo cáo PDF cuối cùng ---
print("\nBắt đầu tạo báo cáo PDF cuối cùng..."); styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Vietnamese', fontName='Roboto', fontSize=10)); styles.add(ParagraphStyle(name='Vietnamese_h1', parent=styles['h1'], fontName='Roboto')); styles.add(ParagraphStyle(name='Vietnamese_h2', parent=styles['h2'], fontName='Roboto'))
story = [Paragraph("BÁO CÁO KẾT QUẢ HUẤN LUYỆN MÔ HÌNH", styles['Vietnamese_h1']), Paragraph(f"Thời gian: {current_time_str} | Mô hình: {CONFIG.MODEL_ID}", styles['Vietnamese']), Spacer(1, 12),
         Paragraph("I. Biểu đồ Huấn luyện & Đánh giá", styles['Vietnamese_h2']), ReportlabImage(metrics_plot_path, width=450, height=550), PageBreak(),
         Paragraph("II. Báo cáo & Ma trận Nhầm lẫn", styles['Vietnamese_h2']), ReportlabImage(cm_plot_path, width=400, height=300), Spacer(1, 12)]
report_df = pd.DataFrame(classification_report(all_true, all_preds, target_names=class_names, output_dict=True)).transpose().round(2).reset_index(); table = Table([report_df.columns.tolist()] + report_df.values.tolist(), colWidths=[80] + [50]*(len(class_names)+3)); table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey), ('GRID', (0,0), (-1,-1), 1, colors.black), ('FONTNAME', (0,0), (-1,-1), 'Roboto')]))
story.extend([table, PageBreak(), Paragraph("III. Phân tích Hiệu suất Nâng cao", styles['Vietnamese_h2']), ReportlabImage(roc_pr_plot_path, width=500, height=350), Spacer(1, 24),
              Paragraph("IV. Trực quan hóa Embedding (t-SNE)", styles['Vietnamese_h2']), ReportlabImage(tsne_plot_path, width=500, height=500), PageBreak(),
              Paragraph("V. Phân tích Lỗi sai & XAI (Mẫu dự đoán SAI)", styles['Vietnamese_h2'])])
for title, path in error_xai_paths:
    if os.path.exists(path): story.extend([Paragraph(title, styles['Vietnamese']), ReportlabImage(path, width=400, height=400), Spacer(1, 12)])

# Thêm mục mới vào PDF
story.append(PageBreak())
story.append(Paragraph("VI. Phân tích XAI trên các mẫu dự đoán ĐÚNG (Grad-CAM Trung bình)", styles['Vietnamese_h2']))
for title, path in avg_cam_paths:
    if os.path.exists(path):
        story.extend([Paragraph(title, styles['Vietnamese']), ReportlabImage(path, width=400, height=400), Spacer(1, 12)])

pdf_report_path = os.path.join(SESSION_FOLDER, f"final_report_{current_time_str}.pdf")
SimpleDocTemplate(pdf_report_path, pagesize=letter).build(story)
print(f"\nBÁO CÁO PDF HOÀN CHỈNH ĐÃ ĐƯỢỢC LƯU TẠI: {pdf_report_path}")
print("\nPIPELINE HOÀN TẤT!")
