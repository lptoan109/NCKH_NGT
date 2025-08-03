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
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
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
    EARLY_STOPPING_PATIENCE = 5

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
    SESSION_FOLDER = os.path.join(BASE_DRIVE_OUTPUT_PATH, f"session_{get_vn_time_str()}")
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
# BLOCK 3: DATASET VÀ DATALOADER
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
        return {"input_values": inputs['input_values'].squeeze(0), "labels": torch.tensor(self.labels[idx], dtype=torch.long)}

all_files, all_labels = [], []
class_names = sorted([d for d in os.listdir(CONFIG.DATA_PATH) if os.path.isdir(os.path.join(CONFIG.DATA_PATH, d))])
class_to_idx = {name: i for i, name in enumerate(class_names)}
idx_to_class = {i: name for i, name in enumerate(class_names)}
for class_name in class_names:
    class_dir = os.path.join(CONFIG.DATA_PATH, class_name)
    for file_name in os.listdir(class_dir):
        if file_name.endswith(".wav"): all_files.append(os.path.join(class_dir, file_name)); all_labels.append(class_to_idx[class_name])
print(f"Tìm thấy {len(all_files)} files trong {len(class_names)} lớp. Các lớp: {class_to_idx}")
train_files, temp_files, train_labels, temp_labels = train_test_split(all_files, all_labels, test_size=0.3, random_state=42, stratify=all_labels)
val_files, test_files, val_labels, test_labels = train_test_split(temp_files, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels)
feature_extractor = AutoFeatureExtractor.from_pretrained(CONFIG.MODEL_ID)
train_dataset, val_dataset, test_dataset = CoughDataset(train_files, train_labels, feature_extractor), CoughDataset(val_files, val_labels, feature_extractor), CoughDataset(test_files, test_labels, feature_extractor)
train_loader, val_loader, test_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True), DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE), DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE)
print("Hoàn tất Block 3.")


# ======================================================================================
# BLOCK 4: MÔ HÌNH VÀ CÁC TIỆN ÍCH NÂNG CAO
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
# BLOCK 5: VÒNG LẶP HUẤN LUYỆN
# ======================================================================================
print("\nBLOCK 5: BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN...")
model = load_model(len(class_names))
optimizer = AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE)
num_training_steps = CONFIG.EPOCHS * len(train_loader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
early_stopper = EarlyStopping(patience=CONFIG.EARLY_STOPPING_PATIENCE, verbose=True, path=os.path.join(CHECKPOINT_FOLDER, "best_model.pth.tar"))
start_epoch, history = load_checkpoint(model, optimizer, lr_scheduler, CHECKPOINT_FOLDER) if CONFIG.RESUME_TRAINING else (0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []})

for epoch in range(start_epoch, CONFIG.EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{CONFIG.EPOCHS} ---")
    model.train()
    total_loss, total_correct, total_samples = 0, 0, 0
    progress_bar = tqdm(train_loader, desc="Training")
    for batch in progress_bar:
        if -1 in batch['labels']: continue
        input_values, labels = batch['input_values'].to(device), batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(input_values, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        predictions = torch.argmax(outputs.logits, dim=-1)
        total_loss += loss.item() * input_values.size(0); total_correct += (predictions == labels).sum().item(); total_samples += labels.size(0)
        progress_bar.set_postfix(loss=total_loss/total_samples, acc=total_correct/total_samples)
    train_loss, train_acc = total_loss / total_samples, total_correct / total_samples
    history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)

    model.eval()
    total_loss, total_correct, total_samples = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            if -1 in batch['labels']: continue
            input_values, labels = batch['input_values'].to(device), batch['labels'].to(device)
            outputs = model(input_values, labels=labels)
            total_loss += outputs.loss.item() * input_values.size(0); total_correct += (torch.argmax(outputs.logits, dim=-1) == labels).sum().item(); total_samples += labels.size(0)
    val_loss, val_acc = total_loss / total_samples, total_correct / total_samples
    history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
    print(f"Epoch {epoch + 1} Summary: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
    
    save_checkpoint({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': lr_scheduler.state_dict(), 'history': history}, CHECKPOINT_FOLDER)
    early_stopper(val_loss, model.state_dict())
    if early_stopper.early_stop: print("Early stopping triggered!"); break
print("\nHoàn tất quá trình huấn luyện.")

# ======================================================================================
# BLOCK 6: PHÂN TÍCH TOÀN DIỆN VÀ BÁO CÁO KẾT QUẢ (PHIÊN BẢN CUỐI CÙNG)
# ======================================================================================
print("\nBLOCK 6: BẮT ĐẦU ĐÁNH GIÁ, PHÂN TÍCH SÂU VÀ TẠO BÁO CÁO...")

# --- 0. Tải lại mô hình tốt nhất để đánh giá ---
best_model_path = os.path.join(CHECKPOINT_FOLDER, "best_model.pth.tar")
if os.path.exists(best_model_path):
    print(f"Tải lại trọng số tốt nhất từ: {best_model_path}")
    # Tải lại model từ đầu để đảm bảo có đúng kiến trúc
    model = load_model(len(class_names))
    model.load_state_dict(torch.load(best_model_path))
else:
    print("Cảnh báo: Không tìm thấy best_model.pth.tar. Sử dụng mô hình cuối cùng để đánh giá.")

current_time_str = get_vn_time_str()
model.eval()

# --- 1. Thu thập tất cả kết quả, xác suất và embedding từ tập Test ---
print("Bắt đầu thu thập kết quả, xác suất và embeddings từ tập Test...")
all_preds, all_true, all_probs, all_embeddings = [], [], [], []
# Tạo một danh sách các file path tương ứng với thứ tự của test_loader
# Điều này rất quan trọng để ánh xạ lỗi sai về đúng file gốc
test_filepaths_in_order = [path for path in test_dataset.file_paths]

with torch.no_grad():
    for batch in tqdm(test_loader, "Collecting Test Set Results"):
        if -1 in batch['labels']: continue
        valid_indices = batch['labels'] != -1
        if not torch.any(valid_indices): continue
        
        input_values = batch['input_values'][valid_indices].to(device)
        labels = batch['labels'][valid_indices]

        # Yêu cầu mô hình trả về cả hidden_states và attentions
        outputs = model(input_values, output_hidden_states=True, output_attentions=True)
        
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)
        
        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(predictions.cpu().numpy())
        all_true.extend(labels.cpu().numpy())
        # Lấy embedding của token [CLS] (vị trí 0) từ lớp ẩn cuối cùng
        all_embeddings.extend(outputs.hidden_states[-1][:, 0, :].cpu().numpy())

all_true, all_preds, all_probs, all_embeddings = np.array(all_true), np.array(all_preds), np.array(all_probs), np.array(all_embeddings)

# --- 2. Tạo và lưu các biểu đồ hiệu suất ---
# Biểu đồ Metrics
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
ax1.plot(history['train_loss'], label='Train Loss'); ax1.plot(history['val_loss'], label='Validation Loss')
ax1.set_title('Training & Validation Loss'); ax1.legend(); ax1.grid(True)
ax2.plot(history['train_acc'], label='Train Accuracy'); ax2.plot(history['val_acc'], label='Validation Accuracy')
ax2.set_title('Training & Validation Accuracy'); ax2.legend(); ax2.grid(True)
metrics_plot_path = os.path.join(SESSION_FOLDER, f"metrics_plot_{current_time_str}.png")
plt.tight_layout(); plt.savefig(metrics_plot_path); plt.close(fig)
print(f"Đã lưu biểu đồ metrics tại: {metrics_plot_path}")

# Ma trận nhầm lẫn
cm = confusion_matrix(all_true, all_preds)
plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names); plt.title('Confusion Matrix')
cm_plot_path = os.path.join(SESSION_FOLDER, f"confusion_matrix_{current_time_str}.png")
plt.tight_layout(); plt.savefig(cm_plot_path); plt.close()
print(f"Đã lưu ma trận nhầm lẫn tại: {cm_plot_path}")

# Biểu đồ ROC/AUC và Precision-Recall
y_true_binarized = label_binarize(all_true, classes=range(len(class_names)))
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8)); colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
for i, color in zip(range(len(class_names)), colors):
    fpr, tpr, _ = roc_curve(y_true_binarized[:, i], all_probs[:, i]); roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, color=color, lw=2, label=f'ROC {class_names[i]} (AUC = {roc_auc:0.2f})')
    precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], all_probs[:, i])
    ax2.plot(recall, precision, color=color, lw=2, label=f'PR {class_names[i]}')
ax1.plot([0, 1], [0, 1], 'k--', lw=2); ax1.set_title('Multi-class ROC Curve'); ax1.legend(loc="lower right"); ax1.grid(True)
ax2.set_title('Multi-class Precision-Recall Curve'); ax2.legend(loc="best"); ax2.grid(True)
roc_pr_plot_path = os.path.join(SESSION_FOLDER, f"roc_pr_plot_{current_time_str}.png")
plt.savefig(roc_pr_plot_path); plt.close(fig)
print(f"Đã lưu biểu đồ ROC & PR tại: {roc_pr_plot_path}")

# --- 3. Trực quan hóa Embedding (t-SNE) ---
print("\nBắt đầu tạo biểu đồ t-SNE...")
tsne = TSNE(n_components=2, verbose=0, perplexity=min(30, len(all_embeddings)-1), n_iter=300, random_state=42)
tsne_results = tsne.fit_transform(all_embeddings)
df_tsne = pd.DataFrame(tsne_results, columns=['tsne-1', 'tsne-2'])
df_tsne['label'] = [idx_to_class[i] for i in all_true]
plt.figure(figsize=(10, 10)); sns.scatterplot(x="tsne-1", y="tsne-2", hue="label", palette=sns.color_palette("hsv", len(class_names)), data=df_tsne, legend="full", alpha=0.8)
plt.title("t-SNE Visualization of Cough Embeddings"); plt.grid(True)
tsne_plot_path = os.path.join(SESSION_FOLDER, f"tsne_plot_{current_time_str}.png")
plt.savefig(tsne_plot_path); plt.close()
print(f"Đã lưu biểu đồ t-SNE tại: {tsne_plot_path}")

# --- 4. Phân tích Lỗi Sai Chuyên sâu và XAI ---
def visualize_single_attention_map(model, file_path, output_path):
    # Tạo lại input tensor cho chỉ một file
    audio_samples = preprocess_audio(file_path)
    if audio_samples is None: return
    inputs = feature_extractor(audio_samples, sampling_rate=CONFIG.SAMPLE_RATE, return_tensors="pt")
    input_tensor = inputs['input_values'].to(device)

    with torch.no_grad():
        outputs = model(input_tensor, output_attentions=True)
    
    attentions = torch.mean(outputs.attentions[-1].squeeze(0), dim=0).cpu().numpy()[1:, 1:]
    grid_size = int(np.sqrt(attentions.shape[0]))
    
    spectrogram_db = librosa.amplitude_to_db(np.abs(librosa.stft(audio_samples)), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4)); librosa.display.specshow(spectrogram_db, sr=CONFIG.SAMPLE_RATE, x_axis='time', y_axis='mel', ax=ax)
    ax.imshow(attentions.reshape(grid_size, grid_size), cmap='viridis', alpha=0.5, aspect='auto', extent=[0, spectrogram_db.shape[1]/100, 0, CONFIG.SAMPLE_RATE/2], origin='lower')
    ax.set_title("Single Sample Attention Map")
    plt.tight_layout(); plt.savefig(output_path); plt.close(fig)

print("\nBắt đầu phân tích các mẫu bị phân loại sai...")
misclassified_indices = np.where(all_preds != all_true)[0]
error_analysis, error_xai_paths = [], []
for idx in misclassified_indices:
    error_analysis.append((all_probs[idx][all_preds[idx]], all_true[idx], all_preds[idx], test_filepaths_in_order[idx]))
error_analysis.sort(key=lambda x: x[0], reverse=True)

print(f"\n--- Top 5 Lỗi sai Tự tin nhất ---")
for i, (confidence, true_label_idx, pred_label_idx, file_path) in enumerate(error_analysis[:5]):
    error_xai_path = os.path.join(SESSION_FOLDER, f"error_analysis_sample_{i+1}_{current_time_str}.png")
    visualize_single_attention_map(model, file_path, error_xai_path)
    error_info = (f"Lỗi #{i+1}: '{os.path.basename(file_path)}' | Thật: '{idx_to_class[true_label_idx]}' | Sai: '{idx_to_class[pred_label_idx]}' (Conf: {confidence:.2f})", error_xai_path)
    error_xai_paths.append(error_info)
    print(error_info[0])
    print(f"  - Đã lưu XAI của lỗi sai tại: {error_xai_path}")

# --- 5. Tạo báo cáo PDF cuối cùng ---
print("\nBắt đầu tạo báo cáo PDF cuối cùng..."); styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='Vietnamese', fontName='Roboto', fontSize=10)); styles.add(ParagraphStyle(name='Vietnamese_h1', parent=styles['h1'], fontName='Roboto')); styles.add(ParagraphStyle(name='Vietnamese_h2', parent=styles['h2'], fontName='Roboto')); styles.add(ParagraphStyle(name='Vietnamese_h3', parent=styles['h3'], fontName='Roboto'))
story = [Paragraph("BÁO CÁO KẾT QUẢ HUẤN LUYỆN MÔ HÌNH", styles['Vietnamese_h1']),
         Paragraph(f"Thời gian: {current_time_str} | Mô hình: {CONFIG.MODEL_ID}", styles['Vietnamese']), Spacer(1, 12),
         Paragraph("I. Biểu đồ Huấn luyện & Đánh giá", styles['Vietnamese_h2']), Image(metrics_plot_path, width=450, height=550), PageBreak(),
         Paragraph("II. Báo cáo & Ma trận Nhầm lẫn", styles['Vietnamese_h2']), Image(cm_plot_path, width=400, height=300), Spacer(1, 12)]
report_df = pd.DataFrame(classification_report(all_true, all_preds, target_names=class_names, output_dict=True)).transpose().round(2).reset_index()
table = Table([report_df.columns.tolist()] + report_df.values.tolist(), colWidths=[80] + [50]*(len(class_names)+3))
table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey), ('GRID', (0,0), (-1,-1), 1, colors.black), ('FONTNAME', (0,0), (-1,-1), 'Roboto')]))
story.append(table); story.append(PageBreak()); story.append(Paragraph("III. Phân tích Hiệu suất Nâng cao", styles['Vietnamese_h2'])); story.append(Image(roc_pr_plot_path, width=500, height=350)); story.append(Spacer(1, 24))
story.append(Paragraph("IV. Trực quan hóa Embedding (t-SNE)", styles['Vietnamese_h2'])); story.append(Image(tsne_plot_path, width=500, height=500)); story.append(PageBreak())
story.append(Paragraph("V. Phân tích Lỗi sai & XAI", styles['Vietnamese_h2']))
for title, path in error_xai_paths:
    if os.path.exists(path):
        story.extend([Paragraph(title, styles['Vietnamese']), Image(path, width=500, height=200), Spacer(1, 12)])
pdf_report_path = os.path.join(SESSION_FOLDER, f"final_report_{current_time_str}.pdf")
SimpleDocTemplate(pdf_report_path, pagesize=letter).build(story)
print(f"\nBÁO CÁO PDF HOÀN CHỈNH ĐÃ ĐƯỢC LƯU TẠI: {pdf_report_path}")
print("\nPIPELINE HOÀN TẤT!")