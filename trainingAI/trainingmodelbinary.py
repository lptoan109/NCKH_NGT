# ======================================================================================
# BLOCK 1: CÀI ĐẶT, IMPORT VÀ CẤU HÌNH NÂNG CAO
# ======================================================================================
print("BLOCK 1: CÀI ĐẶT, IMPORT VÀ CẤU HÌNH NÂNG CAO...")

# Cài đặt các thư viện cần thiết
!pip install transformers torch torchaudio scikit-learn pandas matplotlib seaborn librosa pydub pytz reportlab -q

import os
import sys
import numpy as np
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve, f1_score, precision_score, recall_score
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import librosa.display
from pydub import AudioSegment
from pydub.silence import split_on_silence
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, AdamW, get_scheduler
from google.colab import drive
import datetime
import pytz
from tqdm.auto import tqdm
import requests
from itertools import cycle

# Thư viện cho PDF Report
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Tải phông chữ tiếng Việt cho báo cáo PDF
!wget "https://github.com/google/fonts/raw/main/ofl/roboto/Roboto-Regular.ttf" -O Roboto-Regular.ttf -q
VIETNAMESE_FONT_PATH = "Roboto-Regular.ttf"
pdfmetrics.registerFont(TTFont('Roboto', VIETNAMESE_FONT_PATH))

# Cấu hình toàn bộ pipeline
class CONFIG:
    # --- CÔNG TẮC KHÔI PHỤC ---
    RESUME_TRAINING = False 
    RESUME_SESSION_FOLDER_NAME = "" 
    
    # --- Cấu hình Mô hình & Huấn luyện ---
    MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
    EPOCHS = 50
    LEARNING_RATE = 5e-5
    BATCH_SIZE = 16
    
    # --- Cấu hình Chống Overfitting ---
    EARLY_STOPPING_PATIENCE = 10

    # --- Cấu hình Dữ liệu & Tiền xử lý ---
    DATA_PATH = "/content/data/"
    SAMPLE_RATE = 16000
    MAX_LENGTH_SECS = 10
    SILENCE_THRESH = -40
    MIN_SILENCE_LEN = 300
    
    # --- Cấu hình Google Drive & Output ---
    DRIVE_MOUNT_PATH = "/content/drive"
    DRIVE_OUTPUT_FOLDER_ID = "YOUR_GOOGLE_DRIVE_FOLDER_ID" # !!! THAY ĐỔI ID NÀY !!!
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
    print(f"Chế độ RESUME: Sẽ tiếp tục từ session tại: {SESSION_FOLDER}")
else:
    SESSION_FOLDER = os.path.join(BASE_DRIVE_OUTPUT_PATH, f"session_binary_{get_vn_time_str()}") # Thêm prefix 'binary'
    os.makedirs(SESSION_FOLDER, exist_ok=True)
    print(f"Chế độ MỚI: Output sẽ được lưu tại: {SESSION_FOLDER}")

CHECKPOINT_FOLDER = os.path.join(SESSION_FOLDER, CONFIG.CHECKPOINT_SUBFOLDER)
os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)
print("Hoàn tất Block 1.")


# ======================================================================================
# BLOCK 2: CÁC HÀM TIỀN XỬ LÝ ÂM THANH 
# ======================================================================================
print("\nBLOCK 2: KHỞI TẠO CÁC HÀM TIỀN XỬ LÝ ÂM THANH...")
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
        else: samples = np.concatenate((samples, np.zeros(max_samples - len(samples))))
        return samples
    except Exception as e:
        print(f"Lỗi xử lý file {file_path}: {e}"); return None
print("Hoàn tất Block 2.")


# ======================================================================================
# BLOCK 3: DATASET VÀ DATALOADER (ĐÃ CHỈNH SỬA)
# ======================================================================================
print("\nBLOCK 3: KHỞI TẠO DATASET VÀ DATALOADER...")
class CoughDataset(Dataset):
    def __init__(self, file_paths, labels, feature_extractor):
        self.file_paths, self.labels, self.feature_extractor = file_paths, labels, feature_extractor
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        audio_samples = preprocess_audio(self.file_paths[idx])
        if audio_samples is None: return {"input_values": torch.zeros(1, CONFIG.SAMPLE_RATE * CONFIG.MAX_LENGTH_SECS), "labels": -1}
        inputs = self.feature_extractor(audio_samples, sampling_rate=CONFIG.SAMPLE_RATE, return_tensors="pt")
        
        return {"input_values": inputs['input_values'].squeeze(0), "labels": torch.tensor(self.labels[idx], dtype=torch.float)}

all_files, all_labels = [], []
class_names = sorted([d for d in os.listdir(CONFIG.DATA_PATH) if os.path.isdir(os.path.join(CONFIG.DATA_PATH, d))])
assert len(class_names) == 2, f"Lỗi: Tìm thấy {len(class_names)} lớp. Script này yêu cầu đúng 2 lớp cho bài toán nhị phân."

class_to_idx = {name: i for i, name in enumerate(class_names)}
idx_to_class = {i: name for i, name in enumerate(class_names)}

# Xác định lớp nào là 'ho' - giả định tên thư mục chứa 'cough'
positive_class_name = next((name for name in class_names if 'cough' in name.lower()), None)
assert positive_class_name is not None, "Lỗi: Không tìm thấy thư mục cho lớp 'ho' (cần chứa từ 'cough')."
positive_class_idx = class_to_idx[positive_class_name]
print(f"Lớp dương tính (ho) được xác định là: '{positive_class_name}' (index: {positive_class_idx})")

for class_name in class_names:
    class_dir = os.path.join(CONFIG.DATA_PATH, class_name)
    for file_name in os.listdir(class_dir):
        if file_name.endswith(".wav"): all_files.append(os.path.join(class_dir, file_name)); all_labels.append(class_to_idx[class_name])
print(f"Tìm thấy {len(all_files)} files trong 2 lớp. Các lớp: {class_to_idx}")
train_files, temp_files, train_labels, temp_labels = train_test_split(all_files, all_labels, test_size=0.3, random_state=42, stratify=all_labels)
val_files, test_files, val_labels, test_labels = train_test_split(temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)
feature_extractor = AutoFeatureExtractor.from_pretrained(CONFIG.MODEL_ID)
train_dataset, val_dataset, test_dataset = CoughDataset(train_files, train_labels, feature_extractor), CoughDataset(val_files, val_labels, feature_extractor), CoughDataset(test_files, test_labels, feature_extractor)
train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True), DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE), DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=False)
print("Hoàn tất Block 3.")


# ======================================================================================
# BLOCK 4: MÔ HÌNH VÀ CÁC TIỆN ÍCH NÂNG CAO (ĐÃ CHỈNH SỬA)
# ======================================================================================
print("\nBLOCK 4: KHỞI TẠO MÔ HÌNH VÀ CÁC TIỆN ÍCH...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

def load_model(num_labels):
    return AutoModelForAudioClassification.from_pretrained(
        CONFIG.MODEL_ID, num_labels=num_labels, ignore_mismatched_sizes=True
    ).to(device)

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth.tar'):
        self.patience, self.verbose, self.delta, self.path = patience, verbose, delta, path
        self.counter, self.best_score, self.early_stop, self.val_loss_min = 0, None, False, np.Inf
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
# BLOCK 5: VÒNG LẶP HUẤN LUYỆN (CHO BÀI TOÁN NHỊ PHÂN)
# ======================================================================================
print("\nBLOCK 5: BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN CHO BÀI TOÁN NHỊ PHÂN...")

model = load_model(num_labels=1)
loss_fn = torch.nn.BCEWithLogitsLoss()

optimizer = AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE)
num_training_steps = CONFIG.EPOCHS * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
early_stopper = EarlyStopping(patience=CONFIG.EARLY_STOPPING_PATIENCE, verbose=True, path=os.path.join(CHECKPOINT_FOLDER, "best_model.pth.tar"))
start_epoch, history = load_checkpoint(model, optimizer, lr_scheduler, CHECKPOINT_FOLDER) if CONFIG.RESUME_TRAINING else (0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []})

for epoch in range(start_epoch, CONFIG.EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{CONFIG.EPOCHS} ---")
    
    # --- Training ---
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        if -1 in batch['labels']: continue
        input_values, labels = batch['input_values'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_values)
        logits = outputs.logits
        loss = loss_fn(logits, labels.unsqueeze(1))
        loss.backward()
        optimizer.step(); lr_scheduler.step()
        preds_proba = torch.sigmoid(logits)
        predictions = (preds_proba > 0.5).long().squeeze(1)
        total_loss += loss.item() * input_values.size(0)
        total_correct += (predictions == labels).sum().item()
        total_samples += labels.size(0)
        progress_bar.set_postfix(loss=total_loss/total_samples, acc=total_correct/total_samples)

    train_loss, train_acc = total_loss / total_samples, total_correct / total_samples
    history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)

    # --- Validation ---
    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            if -1 in batch['labels']: continue
            input_values, labels = batch['input_values'].to(device), batch['labels'].to(device)
            outputs = model(input_values)
            logits = outputs.logits
            loss = loss_fn(logits, labels.unsqueeze(1))
            preds_proba = torch.sigmoid(logits)
            predictions = (preds_proba > 0.5).long().squeeze(1)
            total_loss += loss.item() * input_values.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    val_loss, val_acc = total_loss / total_samples, total_correct / total_samples
    history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
    print(f"Epoch {epoch + 1} Summary: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    save_checkpoint({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': lr_scheduler.state_dict(), 'history': history}, CHECKPOINT_FOLDER)
    early_stopper(val_loss, model.state_dict())
    if early_stopper.early_stop: print("Early stopping triggered!"); break
print("\nHoàn tất quá trình huấn luyện.")


# ======================================================================================
# BLOCK 6: PHÂN TÍCH VÀ BÁO CÁO 
# ======================================================================================
print("\nBLOCK 6: BẮT ĐẦU ĐÁNH GIÁ VÀ TẠO BÁO CÁO HOÀN CHỈNH...")

best_model_path = os.path.join(CHECKPOINT_FOLDER, "best_model.pth.tar")
if os.path.exists(best_model_path):
    print(f"Tải lại trọng số tốt nhất từ: {best_model_path}")
    model = load_model(num_labels=1)
    model.load_state_dict(torch.load(best_model_path))
else:
    print("Cảnh báo: Không tìm thấy best_model.pth.tar. Sử dụng mô hình cuối cùng.")

current_time_str = get_vn_time_str()
model.eval()

# --- Hàm tiện ích mới: Lưu DataFrame thành ảnh ---
def save_df_as_image(df, path, title=""):
    fig, ax = plt.subplots(figsize=(8, max(2, len(df) * 0.5))) 
    ax.axis('off'); ax.axis('tight')
    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1.2, 1.2)
    plt.title(title, fontsize=14, pad=20)
    plt.savefig(path, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Đã lưu bảng '{title}' tại: {path}")

# --- 1. Thu thập kết quả từ tập Test ---
print("Bắt đầu thu thập kết quả từ tập Test...")
all_preds_at_50, all_true, all_probs_positive = [], [], []
test_filepaths_in_order = test_files 
with torch.no_grad():
    for batch in tqdm(test_loader, "Collecting Test Set Results"):
        if -1 in batch['labels']: continue
        input_values, labels = batch['input_values'].to(device), batch['labels']
        outputs = model(input_values)
        logits = outputs.logits
        probs = torch.sigmoid(logits).squeeze(1)
        predictions = (probs > 0.5).long()
        all_probs_positive.extend(probs.cpu().numpy())
        all_preds_at_50.extend(predictions.cpu().numpy())
        all_true.extend(labels.cpu().numpy())
all_true, all_preds_at_50, all_probs_positive = np.array(all_true), np.array(all_preds_at_50), np.array(all_probs_positive)

# --- 2. Tạo và lưu các biểu đồ và bảng biểu ---
# ... (Giữ nguyên code vẽ Metrics, CM, Báo cáo, ROC/PR, Threshold từ phiên bản trước) ...
# Biểu đồ Metrics
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
ax1.plot(history['train_loss'], label='Train Loss'); ax1.plot(history['val_loss'], label='Validation Loss'); ax1.set_title('Training & Validation Loss'); ax1.legend(); ax1.grid(True)
ax2.plot(history['train_acc'], label='Train Accuracy'); ax2.plot(history['val_acc'], label='Validation Accuracy'); ax2.set_title('Training & Validation Accuracy'); ax2.legend(); ax2.grid(True)
metrics_plot_path = os.path.join(SESSION_FOLDER, f"metrics_plot_{current_time_str}.png"); plt.tight_layout(); plt.savefig(metrics_plot_path); plt.close(fig)
print(f"Đã lưu biểu đồ metrics tại: {metrics_plot_path}")

# Ma trận nhầm lẫn
cm = confusion_matrix(all_true, all_preds_at_50)
plt.figure(figsize=(6, 5)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names); plt.title('Confusion Matrix (Threshold = 0.5)')
cm_plot_path = os.path.join(SESSION_FOLDER, f"confusion_matrix_{current_time_str}.png"); plt.tight_layout(); plt.savefig(cm_plot_path); plt.close()
print(f"Đã lưu ma trận nhầm lẫn tại: {cm_plot_path}")

# Bảng Báo cáo Phân loại
report_data = classification_report(all_true, all_preds_at_50, target_names=class_names, output_dict=True)
report_df = pd.DataFrame(report_data).transpose().round(3)
report_df.reset_index(inplace=True); report_df.rename(columns={'index': 'Class'}, inplace=True)
report_table_path = os.path.join(SESSION_FOLDER, f"classification_report_{current_time_str}.png")
save_df_as_image(report_df, report_table_path, title="Classification Report (Threshold = 0.5)")

# Biểu đồ ROC/AUC và Precision-Recall
fpr, tpr, _ = roc_curve(all_true, all_probs_positive); roc_auc = auc(fpr, tpr)
precision, recall, _ = precision_recall_curve(all_true, all_probs_positive)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:0.3f})'); ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); ax1.set_title('ROC Curve'); ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate'); ax1.legend(loc="lower right"); ax1.grid(True)
ax2.plot(recall, precision, color='blue', lw=2, label='Precision-Recall curve'); ax2.set_title('Precision-Recall Curve'); ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.legend(loc="best"); ax2.grid(True)
roc_pr_plot_path = os.path.join(SESSION_FOLDER, f"roc_pr_plot_{current_time_str}.png"); plt.savefig(roc_pr_plot_path); plt.close(fig)
print(f"Đã lưu biểu đồ ROC & PR tại: {roc_pr_plot_path}")

# Phân tích Ngưỡng Quyết định
thresholds = np.linspace(0.01, 0.99, 100)
f1_scores, precision_scores, recall_scores = [], [], []
for thresh in thresholds:
    preds = (all_probs_positive > thresh).astype(int)
    f1_scores.append(f1_score(all_true, preds, zero_division=0)); precision_scores.append(precision_score(all_true, preds, zero_division=0)); recall_scores.append(recall_score(all_true, preds, zero_division=0))
optimal_idx = np.argmax(f1_scores); optimal_threshold = thresholds[optimal_idx]
plt.figure(figsize=(10, 6)); plt.plot(thresholds, f1_scores, label='F1-Score', lw=2); plt.plot(thresholds, precision_scores, label='Precision', linestyle=':'); plt.plot(thresholds, recall_scores, label='Recall', linestyle=':')
plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold (F1-max) = {optimal_threshold:.2f}')
plt.title('Threshold Analysis'); plt.xlabel('Threshold'); plt.ylabel('Score'); plt.legend(); plt.grid(True)
threshold_plot_path = os.path.join(SESSION_FOLDER, f"threshold_analysis_{current_time_str}.png"); plt.savefig(threshold_plot_path); plt.close()
print(f"Đã lưu biểu đồ phân tích ngưỡng tại: {threshold_plot_path}")


# --- 3. Phân tích Lỗi Sai và XAI Chuyên sâu cho Lớp "Ho" ---
print("\nBắt đầu phân tích XAI chuyên sâu...")
# Hàm visualize attention cho một mẫu duy nhất
def visualize_single_attention_map(model, file_path, output_path, title=""):
    audio_samples = preprocess_audio(file_path);
    if audio_samples is None: return
    inputs = feature_extractor(audio_samples, sampling_rate=CONFIG.SAMPLE_RATE, return_tensors="pt")
    input_tensor = inputs['input_values'].to(device)
    with torch.no_grad(): outputs = model(input_tensor, output_attentions=True)
    attentions = torch.mean(outputs.attentions[-1].squeeze(0), dim=0).cpu().numpy()[1:, 1:]
    grid_size = int(np.sqrt(attentions.shape[0]))
    spectrogram_db = librosa.amplitude_to_db(np.abs(librosa.stft(audio_samples)), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4)); librosa.display.specshow(spectrogram_db, sr=CONFIG.SAMPLE_RATE, x_axis='time', y_axis='mel', ax=ax)
    ax.imshow(attentions.reshape(grid_size, grid_size), cmap='viridis', alpha=0.5, aspect='auto', extent=[0, spectrogram_db.shape[1]/100, 0, CONFIG.SAMPLE_RATE/2], origin='lower')
    ax.set_title(title); plt.tight_layout(); plt.savefig(output_path); plt.close(fig)

# --- Phân tích Lỗi Sai ---
fp_indices = np.where((all_preds_at_50 == 1) & (all_true == 0))[0]
fn_indices = np.where((all_preds_at_50 == 0) & (all_true == 1))[0]
fp_errors, fn_errors = [], []
for idx in fp_indices: fp_errors.append((all_probs_positive[idx], test_filepaths_in_order[idx]))
for idx in fn_indices: fn_errors.append((1 - all_probs_positive[idx], test_filepaths_in_order[idx]))
fp_errors.sort(key=lambda x: x[0], reverse=True); fn_errors.sort(key=lambda x: x[0], reverse=True)
error_list_for_df = []
for i, (confidence, file_path) in enumerate(fp_errors[:5]): error_list_for_df.append([f"FP #{i+1}", os.path.basename(file_path), f"{confidence:.3f}"])
for i, (confidence, file_path) in enumerate(fn_errors[:5]): error_list_for_df.append([f"FN #{i+1}", os.path.basename(file_path), f"{1-confidence:.3f}"])
error_df = pd.DataFrame(error_list_for_df, columns=["Error Type", "File Name", "Confidence"])
error_table_path = os.path.join(SESSION_FOLDER, f"error_analysis_table_{current_time_str}.png")
if not error_df.empty: save_df_as_image(error_df, error_table_path, title="Top Misclassified Samples")

# --- Phân tích XAI cho lớp "Ho" ---
# 1. Bản đồ chú ý trung bình cho lớp "Ho"
correct_cough_indices = np.where((all_preds_at_50 == positive_class_idx) & (all_true == positive_class_idx))[0]
avg_attention_list = []
print(f"\nTạo attention map trung bình cho lớp '{positive_class_name}'...")
for idx in tqdm(correct_cough_indices, desc="Averaging Attention"):
    input_tensor = test_dataset[idx]['input_values'].unsqueeze(0).to(device)
    with torch.no_grad(): outputs = model(input_tensor, output_attentions=True)
    attentions = torch.mean(outputs.attentions[-1].squeeze(0), dim=0).cpu().numpy()[1:, 1:]
    avg_attention_list.append(attentions)
if avg_attention_list:
    avg_attention = np.mean(np.stack(avg_attention_list), axis=0)
    grid_size = int(np.sqrt(avg_attention.shape[0]))
    fig, ax = plt.subplots(figsize=(10, 4)); librosa.display.specshow(np.zeros((128, 1000)), sr=CONFIG.SAMPLE_RATE, x_axis='time', y_axis='mel', ax=ax)
    ax.imshow(avg_attention.reshape(grid_size, grid_size), cmap='viridis', aspect='auto', extent=[0, CONFIG.MAX_LENGTH_SECS, 0, CONFIG.SAMPLE_RATE/2])
    ax.set_title(f"Average Attention Map for '{positive_class_name}' class")
    avg_attention_path = os.path.join(SESSION_FOLDER, f"xai_avg_attention_cough_{current_time_str}.png")
    plt.tight_layout(); plt.savefig(avg_attention_path); plt.close(fig)
    print(f"Đã lưu XAI map trung bình cho lớp Ho tại: {avg_attention_path}")

# 2. Phân tích Top 5 mẫu "Ho" được nhận diện đúng nhất
correct_cough_analysis = []
for idx in correct_cough_indices: correct_cough_analysis.append((all_probs_positive[idx], test_filepaths_in_order[idx]))
correct_cough_analysis.sort(key=lambda x: x[0], reverse=True)
top_correct_list_df = []
top_correct_xai_paths = []
print(f"\n--- Top 5 Mẫu Ho Đúng Nhất (Most Confident Correct) ---")
for i, (confidence, file_path) in enumerate(correct_cough_analysis[:5]):
    top_correct_list_df.append([f"Top Correct #{i+1}", os.path.basename(file_path), f"{confidence:.3f}"])
    xai_path = os.path.join(SESSION_FOLDER, f"xai_top_correct_sample_{i+1}_{current_time_str}.png")
    visualize_single_attention_map(model, test_files.index(file_path), file_path, xai_path, title=f"Top Correct #{i+1}")
    top_correct_xai_paths.append((f"Top Correct #{i+1}: '{os.path.basename(file_path)}' (Conf: {confidence:.3f})", xai_path))
    print(f"  - Đã lưu XAI cho mẫu Ho đúng tại: {xai_path}")
top_correct_df = pd.DataFrame(top_correct_list_df, columns=["Rank", "File Name", "Confidence"])
top_correct_table_path = os.path.join(SESSION_FOLDER, f"top_correct_table_{current_time_str}.png")
if not top_correct_df.empty: save_df_as_image(top_correct_df, top_correct_table_path, title="Top Correctly Classified 'Cough' Samples")


# --- 4. Tạo báo cáo PDF cuối cùng ---
print("\nBắt đầu tạo báo cáo PDF cuối cùng..."); styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Vietnamese', fontName='Roboto', fontSize=10)); styles.add(ParagraphStyle(name='Vietnamese_h1', parent=styles['h1'], fontName='Roboto')); styles.add(ParagraphStyle(name='Vietnamese_h2', parent=styles['h2'], fontName='Roboto')); styles.add(ParagraphStyle(name='Vietnamese_h3', parent=styles['h3'], fontName='Roboto'))
story = [Paragraph("BÁO CÁO KẾT QUẢ - MÔ HÌNH PHÁT HIỆN HO", styles['Vietnamese_h1']),
         Paragraph(f"Thời gian: {current_time_str} | Mô hình: {CONFIG.MODEL_ID}", styles['Vietnamese']), Spacer(1, 12),
         Paragraph("I. Biểu đồ Huấn luyện", styles['Vietnamese_h2']), Image(metrics_plot_path, width=450, height=550), PageBreak(),
         Paragraph("II. Báo cáo & Ma trận Nhầm lẫn (Ngưỡng 0.5)", styles['Vietnamese_h2']), Image(report_table_path, width=500), Spacer(1, 12), Image(cm_plot_path, width=400, height=300), PageBreak(),
         Paragraph("III. Phân tích Hiệu suất Nâng cao", styles['Vietnamese_h2']), Image(roc_pr_plot_path, width=500, height=250), Spacer(1, 12),
         Paragraph("IV. Phân tích Ngưỡng Quyết định", styles['Vietnamese_h2']), Image(threshold_plot_path, width=500, height=300), PageBreak(),
         Paragraph("V. Phân tích Lỗi sai (Top 5)", styles['Vietnamese_h2']), Image(error_table_path, width=500), PageBreak(),
         Paragraph("VI. Phân tích XAI Chuyên sâu Lớp 'Ho'", styles['Vietnamese_h2']),
         Paragraph("1. Bản đồ Chú ý Trung bình", styles['Vietnamese_h3']), Image(avg_attention_path, width=500, height=200), Spacer(1, 12),
         Paragraph("2. Top 5 Mẫu 'Ho' được Nhận diện Đúng nhất", styles['Vietnamese_h3']), Image(top_correct_table_path, width=500)]
for title, path in top_correct_xai_paths:
    if os.path.exists(path): story.extend([Paragraph(title, styles['Vietnamese']), Image(path, width=500, height=200), Spacer(1, 12)])
pdf_report_path = os.path.join(SESSION_FOLDER, f"final_report_binary_{current_time_str}.pdf")
SimpleDocTemplate(pdf_report_path, pagesize=letter).build(story)
print(f"\nBÁO CÁO PDF HOÀN CHỈNH ĐÃ ĐƯỢC LƯU TẠI: {pdf_report_path}")
print("\nPIPELINE HOÀN TẤT!")