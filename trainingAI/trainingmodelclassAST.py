# ======================================================================================
# BLOCK 1: CÀI ĐẶT, IMPORT VÀ CẤU HÌNH
# ======================================================================================
print("BLOCK 1: CÀI ĐẶT, IMPORT VÀ CẤU HÌNH...")

# Cài đặt các thư viện cần thiết
!pip install -q timm torch torchaudio scikit-learn pandas matplotlib seaborn librosa pydub pytz reportlab grad-cam transformers noisereduce huggingface_hub

# --- Imports ---
import os
import sys
import random
import shutil
import datetime
from itertools import cycle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T
from torchvision import transforms
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from pydub import AudioSegment
from pydub.silence import split_on_silence
from torch.optim import AdamW
from transformers import get_scheduler, AutoFeatureExtractor, AutoModelForAudioClassification
from google.colab import drive
import pytz
from tqdm.auto import tqdm
from PIL import Image
import noisereduce as nr
from torch.cuda.amp import autocast, GradScaler
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as ReportlabImage, PageBreak
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


# --- Cấu hình toàn bộ pipeline ---
class CONFIG:
    # --- CÔNG TẮC ĐIỀU KHIỂN ---
    RUN_PREPROCESSING = True
    RESUME_TRAINING = False
    RESUME_SESSION_FOLDER_NAME = ""

    # --- Cấu hình Mô hình & Huấn luyện ---
    MODEL_ID = "MIT/ast-finetuned-audioset-10-10-fe" # Audio Spectrogram Transformer
    EPOCHS = 50
    LEARNING_RATE = 5e-5 # Tốc độ học phù hợp cho fine-tuning Transformer
    BATCH_SIZE = 32 # Giảm batch size vì AST lớn hơn ResNet
    SEED = 42

    # --- Cấu hình Chống Overfitting ---
    EARLY_STOPPING_PATIENCE = 10
    USE_DATA_AUGMENTATION = False
    EARLY_STOPPING_METRIC = 'val_f1_score' # 'val_loss' hoặc 'val_f1_score'

    # --- Cấu hình Dữ liệu & Đường dẫn ---
    CLASSES_TO_USE = ["covid", "tuberculosis"] # CHỈ ĐỊNH CÁC LỚP CHO BÀI TOÁN NHỊ PHÂN
    DRIVE_MOUNT_PATH = "/content/drive"
    DATA_PATH = "/content/drive/MyDrive/Tai_Lieu_NCKH/dataset_cough"
    PREPROCESSED_DATA_PATH = "Tai_Lieu_NCKH/AST_covid_tuberculosis_data" # Thư mục mới cho dữ liệu nhị phân
    DRIVE_OUTPUT_PATH = "Tai_Lieu_NCKH/newAiData/AST_coid-tuberculosis" # Thư mục output mới
    CHECKPOINT_SUBFOLDER = "checkpoints"

    # --- Các cấu hình xử lý âm thanh ---
    SAMPLE_RATE = 16000
    MAX_LENGTH_SECS = 5
    N_MELS = 128 # AST yêu cầu 128 mels
    N_FFT = 2048
    HOP_LENGTH = 512
    SILENCE_THRESH = -40
    MIN_SILENCE_LEN = 300
    
    # --- Cấu hình khác ---
    TIMEZONE = "Asia/Ho_Chi_Minh"

# --- Các hàm tiện ích & Thiết lập môi trường ---
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
set_seed(CONFIG.SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Sử dụng thiết bị: {device}")

drive.mount(CONFIG.DRIVE_MOUNT_PATH)

BASE_DRIVE_OUTPUT_PATH = os.path.join(CONFIG.DRIVE_MOUNT_PATH, "MyDrive", CONFIG.DRIVE_OUTPUT_PATH)
PREPROCESSED_FOLDER_ON_DRIVE = os.path.join(CONFIG.DRIVE_MOUNT_PATH, "MyDrive", CONFIG.PREPROCESSED_DATA_PATH)
os.makedirs(BASE_DRIVE_OUTPUT_PATH, exist_ok=True)
os.makedirs(PREPROCESSED_FOLDER_ON_DRIVE, exist_ok=True)

tz = pytz.timezone(CONFIG.TIMEZONE)
def get_vn_time_str():
    return datetime.datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")

# Cấu hình font cho PDF
try:
    pdfmetrics.registerFont(TTFont('DejaVuSans', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'))
    PDF_FONT_NAME = 'DejaVuSans'
except:
    print("Font DejaVuSans không tồn tại, sử dụng Helvetica.")
    PDF_FONT_NAME = 'Helvetica'

print("Hoàn tất Block 1.")


# ======================================================================================
# BLOCK 2: CÁC HÀM TIỀN XỬ LÝ ÂM THANH
# ======================================================================================
print("\nBLOCK 2: KHỞI TẠO CÁC HÀM TIỀN XỬ LÝ...")

def preprocess_audio(file_path, config):
    try:
        samples, sr = librosa.load(file_path, sr=config.SAMPLE_RATE, res_type='kaiser_fast')
        samples_denoised = nr.reduce_noise(y=samples, sr=sr, verbose=False)
        
        # Cắt khoảng lặng
        int_samples = (samples_denoised * 32767).astype(np.int16)
        sound = AudioSegment(int_samples.tobytes(), frame_rate=sr, sample_width=int_samples.dtype.itemsize, channels=1)
        chunks = split_on_silence(sound, min_silence_len=config.MIN_SILENCE_LEN, silence_thresh=config.SILENCE_THRESH)
        processed_sound = sum(chunks, AudioSegment.empty()) if chunks else sound
        final_samples = np.array(processed_sound.get_array_of_samples()).astype(np.float32) / 32767.0

        # Chuẩn hóa độ dài
        max_samples = config.SAMPLE_RATE * config.MAX_LENGTH_SECS
        if len(final_samples) > max_samples:
            final_samples = final_samples[:max_samples]
        else:
            final_samples = np.pad(final_samples, (0, max_samples - len(final_samples)), 'constant')
        return final_samples
    except Exception as e:
        print(f"Lỗi xử lý file {file_path}: {e}")
        return None

def waveform_to_mel_spectrogram(waveform, config):
    # Tạo Mel-spectrogram tiêu chuẩn (1 kênh) cho AST
    mel_spec = librosa.feature.melspectrogram(
        y=waveform, 
        sr=config.SAMPLE_RATE, 
        n_fft=config.N_FFT, 
        hop_length=config.HOP_LENGTH, 
        n_mels=config.N_MELS
    )
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Chuẩn hóa về [0, 1]
    if log_mel_spec.max() == log_mel_spec.min():
        return np.zeros_like(log_mel_spec)
    normalized_spec = (log_mel_spec - log_mel_spec.min()) / (log_mel_spec.max() - log_mel_spec.min())
    return normalized_spec

print("Hoàn tất Block 2.")


# ======================================================================================
# --- BLOCK CHÍNH: LỰA CHỌN CHẾ ĐỘ CHẠY ---
# ======================================================================================
if CONFIG.RUN_PREPROCESSING:
    # --- CHẾ ĐỘ 1: TIỀN XỬ LÝ DỮ LIỆU ---
    print("\n========================================================================")
    print("                  CHẾ ĐỘ: TIỀN XỬ LÝ DỮ LIỆU NHỊ PHÂN")
    print("========================================================================\n")

    TEMP_PROCESSING_DIR = "/content/temp_processing"
    if os.path.exists(TEMP_PROCESSING_DIR): shutil.rmtree(TEMP_PROCESSING_DIR)
    os.makedirs(TEMP_PROCESSING_DIR)

    print(f"Đang quét dữ liệu cho các lớp được chọn: {CONFIG.CLASSES_TO_USE}")
    all_data = []
    class_names = sorted(CONFIG.CLASSES_TO_USE)
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(CONFIG.DATA_PATH, class_name)
        if not os.path.isdir(class_dir):
            print(f"Cảnh báo: Không tìm thấy thư mục cho lớp '{class_name}'. Bỏ qua.")
            continue
        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith((".wav", ".mp3", ".flac")):
                file_path = os.path.join(class_dir, file_name)
                participant_id = file_name.split('_')[0]
                all_data.append({"file_path": file_path, "label": class_to_idx[class_name], "participant_id": participant_id})
    df = pd.DataFrame(all_data)

    print(f"Đã quét {len(df)} files từ {df['participant_id'].nunique()} cá nhân duy nhất.")
    print("Thực hiện chia dữ liệu theo nhóm (Group Split)...")
    splitter = GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=CONFIG.SEED)
    train_idx, temp_idx = next(splitter.split(df, groups=df['participant_id']))
    train_df, temp_df = df.iloc[train_idx].copy(), df.iloc[temp_idx].copy()
    splitter_val_test = GroupShuffleSplit(test_size=0.5, n_splits=1, random_state=CONFIG.SEED)
    val_idx, test_idx = next(splitter_val_test.split(temp_df, groups=temp_df['participant_id']))
    val_df, test_df = temp_df.iloc[val_idx].copy(), temp_df.iloc[test_idx].copy()
    print(f"Chia dữ liệu hoàn tất: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test.")

    def process_and_save_features(dataframe, set_name, base_output_dir):
        print(f"\nBắt đầu xử lý tập {set_name}...")
        new_rows = []
        for _, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Processing {set_name}"):
            waveform = preprocess_audio(row['file_path'], CONFIG)
            if waveform is not None:
                features = waveform_to_mel_spectrogram(waveform, CONFIG)
                class_name = class_names[row['label']]
                output_class_dir = os.path.join(base_output_dir, set_name, class_name)
                os.makedirs(output_class_dir, exist_ok=True)
                filename = os.path.splitext(os.path.basename(row['file_path']))[0] + ".npy"
                output_npy_path = os.path.join(output_class_dir, filename)
                np.save(output_npy_path, features)
                new_row = row.copy()
                new_row['spectrogram_path'] = os.path.join(set_name, class_name, filename)
                new_rows.append(new_row)
        return pd.DataFrame(new_rows)

    train_df_processed = process_and_save_features(train_df, 'train', TEMP_PROCESSING_DIR)
    val_df_processed = process_and_save_features(val_df, 'val', TEMP_PROCESSING_DIR)
    test_df_processed = process_and_save_features(test_df, 'test', TEMP_PROCESSING_DIR)

    train_df_processed.to_csv(os.path.join(TEMP_PROCESSING_DIR, "train.csv"), index=False)
    val_df_processed.to_csv(os.path.join(TEMP_PROCESSING_DIR, "val.csv"), index=False)
    test_df_processed.to_csv(os.path.join(TEMP_PROCESSING_DIR, "test.csv"), index=False)

    print("\n--- TIỀN XỬ LÝ HOÀN TẤT TRÊN COLAB ---")
    print(f"Bắt đầu sao chép thư mục vào Google Drive tại: {PREPROCESSED_FOLDER_ON_DRIVE}")
    if os.path.exists(PREPROCESSED_FOLDER_ON_DRIVE):
        print("Thư mục đích đã tồn tại. Sẽ xóa và sao chép lại.")
        shutil.rmtree(PREPROCESSED_FOLDER_ON_DRIVE)
    shutil.copytree(TEMP_PROCESSING_DIR, PREPROCESSED_FOLDER_ON_DRIVE)
    print("Sao chép thành công.")
    print("\n💡 HƯỚNG DẪN: Bây giờ hãy đặt CONFIG.RUN_PREPROCESSING = False và chạy lại script để huấn luyện.")

else:
    # --- CHẾ ĐỘ 2: HUẤN LUYỆN VÀ ĐÁNH GIÁ ---
    print("\n========================================================================")
    print("              CHẾ ĐỘ: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH")
    print("========================================================================\n")

    if CONFIG.RESUME_TRAINING and CONFIG.RESUME_SESSION_FOLDER_NAME:
        SESSION_FOLDER = os.path.join(BASE_DRIVE_OUTPUT_PATH, CONFIG.RESUME_SESSION_FOLDER_NAME)
        print(f"Chế độ RESUME: Tiếp tục từ session tại: {SESSION_FOLDER}")
    else:
        SESSION_FOLDER = os.path.join(BASE_DRIVE_OUTPUT_PATH, f"session_{get_vn_time_str()}")
        os.makedirs(SESSION_FOLDER, exist_ok=True)
        print(f"Chế độ MỚI: Output sẽ được lưu tại: {SESSION_FOLDER}")
    
    CHECKPOINT_FOLDER = os.path.join(SESSION_FOLDER, CONFIG.CHECKPOINT_SUBFOLDER)
    os.makedirs(CHECKPOINT_FOLDER, exist_ok=True)

    # ======================================================================================
    # BLOCK 3: DATASET VÀ DATALOADER
    # ======================================================================================
    print("\nBLOCK 3: KHỞI TẠO DATASET VÀ DATALOADER...")
    
    feature_extractor = AutoFeatureExtractor.from_pretrained(CONFIG.MODEL_ID)
    
    class AudioAugmentation:
        def __init__(self, use_augmentation=True):
            self.use_augmentation = use_augmentation
            if use_augmentation:
                self.transforms = torch.nn.Sequential(
                    T.FrequencyMasking(freq_mask_param=48),
                    T.TimeMasking(time_mask_param=48)
                )
        def __call__(self, spec):
            if self.use_augmentation:
                return self.transforms(spec)
            return spec

    train_augmentation = AudioAugmentation(use_augmentation=CONFIG.USE_DATA_AUGMENTATION)
    
    print("Đang tải thông tin các tập dữ liệu từ file CSV...")
    try:
        train_df = pd.read_csv(os.path.join(PREPROCESSED_FOLDER_ON_DRIVE, "train.csv"))
        val_df = pd.read_csv(os.path.join(PREPROCESSED_FOLDER_ON_DRIVE, "val.csv"))
        test_df = pd.read_csv(os.path.join(PREPROCESSED_FOLDER_ON_DRIVE, "test.csv"))
    except FileNotFoundError:
        sys.exit("LỖI: Không tìm thấy file CSV. Hãy chạy chế độ tiền xử lý trước.")
    
    class_names = sorted(CONFIG.CLASSES_TO_USE)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}
    print(f"Đã xác định các lớp cho huấn luyện: {class_names}")

    class CoughDataset(Dataset):
        def __init__(self, df, base_path, feature_extractor, augment_fn=None):
            self.df = df.dropna(subset=['label']).copy()
            self.df['label'] = self.df['label'].astype(int)
            self.base_path = base_path
            self.feature_extractor = feature_extractor
            self.augment_fn = augment_fn

        def __len__(self):
            return len(self.df)
            
        def __getitem__(self, idx):
            row = self.df.iloc[idx]
            path = os.path.join(self.base_path, row['spectrogram_path'])
            try:
                # Tải spectrogram và chuyển thành tensor
                spec = torch.from_numpy(np.load(path)).float()
                
                # Áp dụng augmentation nếu có
                if self.augment_fn:
                    spec = self.augment_fn(spec.unsqueeze(0)).squeeze(0)
                
                # Chuẩn hóa bằng feature extractor của AST
                processed_input = self.feature_extractor(
                    spec.numpy(),
                    sampling_rate=self.feature_extractor.sampling_rate,
                    return_tensors="pt"
                )
                
                return {
                    "input_values": processed_input['input_values'].squeeze(0),
                    "labels": torch.tensor(row['label'], dtype=torch.long),
                    "original_path": row['file_path']
                }
            except Exception as e:
                print(f"Lỗi tải file {path}: {e}")
                return None # Sẽ được lọc ra bởi collate_fn

    def collate_fn(batch):
        # Lọc bỏ các mẫu bị lỗi (None)
        batch = [item for item in batch if item is not None]
        if not batch:
            return None
        return torch.utils.data.default_collate(batch)
    
    print("\nĐang tính toán Class Weights...")
    y_train = train_df.dropna(subset=['label'])['label'].astype(int)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    for i, weight in enumerate(class_weights):
        print(f" - Lớp '{idx_to_class[i]}' (id {i}): weight = {weight:.2f}")

    train_dataset = CoughDataset(train_df, PREPROCESSED_FOLDER_ON_DRIVE, feature_extractor, train_augmentation)
    val_dataset = CoughDataset(val_df, PREPROCESSED_FOLDER_ON_DRIVE, feature_extractor)
    test_dataset = CoughDataset(test_df, PREPROCESSED_FOLDER_ON_DRIVE, feature_extractor)

    train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, num_workers=2, pin_memory=True, collate_fn=collate_fn)
    print("\nHoàn tất Block 3.")


    # ======================================================================================
    # BLOCK 4: MÔ HÌNH VÀ CÁC TIỆN ÍCH
    # ======================================================================================
    print("\nBLOCK 4: KHỞI TẠO MÔ HÌNH VÀ CÁC TIỆN ÍCH...")
    def load_model(num_labels):
        model = AutoModelForAudioClassification.from_pretrained(
            CONFIG.MODEL_ID,
            num_labels=num_labels,
            ignore_mismatched_sizes=True # Rất quan trọng: cho phép thay thế lớp classifier
        )
        return model.to(device)

    class EarlyStopping:
        def __init__(self, patience=5, verbose=False, delta=0, path='c.pth.tar', mode='min', metric_name='val_loss'):
            self.patience, self.verbose, self.delta, self.path = patience, verbose, delta, path
            self.mode, self.metric_name = mode, metric_name
            self.counter, self.best_score, self.early_stop = 0, None, False
            self.val_score_best = np.inf if mode == 'min' else -np.inf

        def __call__(self, score, model_state):
            is_better = (score < self.val_score_best - self.delta) if self.mode == 'min' else (score > self.val_score_best + self.delta)
            if self.best_score is None or is_better:
                self.save_checkpoint(score, model_state)
                self.counter = 0
            else:
                self.counter += 1
                if self.verbose: print(f'EarlyStopping counter: {self.counter}/{self.patience}')
                if self.counter >= self.patience: self.early_stop = True

        def save_checkpoint(self, score, model_state):
            if self.verbose: print(f'{self.metric_name} cải thiện ({self.val_score_best:.6f}-->{score:.6f}). Đang lưu model...')
            torch.save(model_state, self.path)
            self.best_score = self.val_score_best = score
            
    def load_checkpoint(model, optimizer, scheduler, folder):
        filepath = os.path.join(folder, "last_checkpoint.pth.tar")
        if os.path.exists(filepath):
            print(f"Tải checkpoint từ {filepath}")
            ckpt = torch.load(filepath, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            return ckpt['epoch'] + 1, ckpt['history']
        return 0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        
    def save_checkpoint_state(state, folder):
        torch.save(state, os.path.join(folder, "last_checkpoint.pth.tar"))

    print("Hoàn tất Block 4.")


    # ======================================================================================
    # BLOCK 5: VÒNG LẶP HUẤN LUYỆN
    # ======================================================================================
    print("\nBLOCK 5: BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN...")
    model = load_model(len(class_names))
    optimizer = AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    num_training_steps = CONFIG.EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1 * num_training_steps), num_training_steps=num_training_steps)
    early_stopper = EarlyStopping(patience=CONFIG.EARLY_STOPPING_PATIENCE, verbose=True, path=os.path.join(CHECKPOINT_FOLDER, "best_model.pth.tar"), mode='max' if CONFIG.EARLY_STOPPING_METRIC == 'val_f1_score' else 'min', metric_name=CONFIG.EARLY_STOPPING_METRIC)
    start_epoch, history = load_checkpoint(model, optimizer, lr_scheduler, CHECKPOINT_FOLDER) if CONFIG.RESUME_TRAINING else (0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_f1': []})
    scaler = GradScaler()
    
    for epoch in range(start_epoch, CONFIG.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG.EPOCHS} ---")
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            if batch is None: continue
            inputs, labels = batch['input_values'].to(device), batch['labels'].to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type):
                outputs = model(inputs)
                loss = criterion(outputs.logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            predictions = torch.argmax(outputs.logits, dim=-1)
            total_loss += loss.item() * inputs.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            if total_samples > 0: progress_bar.set_postfix(loss=total_loss/total_samples, acc=total_correct/total_samples)
        
        train_loss, train_acc = (total_loss/total_samples, total_correct/total_samples) if total_samples > 0 else (0,0)
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)

        model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        all_val_preds, all_val_true = [], []
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if batch is None: continue
                inputs, labels = batch['input_values'].to(device), batch['labels'].to(device)
                with autocast(device_type=device.type):
                    outputs = model(inputs)
                    loss = criterion(outputs.logits, labels)
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_val_preds.extend(predictions.cpu().numpy()); all_val_true.extend(labels.cpu().numpy())
                total_loss += loss.item() * inputs.size(0)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        val_loss, val_acc = (total_loss/total_samples, total_correct/total_samples) if total_samples > 0 else (0,0)
        val_f1 = f1_score(all_val_true, all_val_preds, average='macro', zero_division=0)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc); history['val_f1'].append(val_f1)
        print(f"Epoch {epoch + 1} Summary: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        score_to_check = val_f1 if CONFIG.EARLY_STOPPING_METRIC == 'val_f1_score' else val_loss
        checkpoint_state = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': lr_scheduler.state_dict(), 'history': history}
        save_checkpoint_state(checkpoint_state, CHECKPOINT_FOLDER)
        early_stopper(score_to_check, model.state_dict())
        if early_stopper.early_stop: print("Early stopping triggered!"); break
    print("\nHoàn tất quá trình huấn luyện.")


    # ======================================================================================
    # BLOCK 6: PHÂN TÍCH TOÀN DIỆN VÀ BÁO CÁO KẾT QUẢ
    # ======================================================================================
    print("\nBLOCK 6: BẮT ĐẦU ĐÁNH GIÁ, PHÂN TÍCH SÂU VÀ TẠO BÁO CÁO...")
    best_model_path = os.path.join(CHECKPOINT_FOLDER, "best_model.pth.tar")
    if os.path.exists(best_model_path):
        print(f"Tải lại trọng số tốt nhất từ: {best_model_path}")
        model = load_model(len(class_names))
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Cảnh báo: Không tìm thấy best_model.pth.tar. Sử dụng mô hình cuối cùng để đánh giá.")
    
    current_time_str = get_vn_time_str()
    model.eval()

    print("Bắt đầu thu thập kết quả từ tập Test...")
    all_preds, all_true, all_probs, all_original_paths = [], [], [], []
    with torch.no_grad():
        for batch in tqdm(test_loader, "Collecting Test Set Results"):
            if batch is None: continue
            inputs, labels = batch['input_values'].to(device), batch['labels'].to(device)
            original_paths = batch['original_path']
            outputs = model(inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())
            all_true.extend(labels.cpu().numpy())
            all_original_paths.extend(original_paths)

    all_true, all_preds, all_probs = np.array(all_true), np.array(all_preds), np.array(all_probs)
    
    print("\nĐang tạo và lưu các biểu đồ hiệu suất...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    ax1.plot(history['train_loss'], label='Train Loss'); ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_title('Training & Validation Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(history['train_acc'], label='Train Accuracy'); ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_title('Training & Validation Accuracy'); ax2.legend(); ax2.grid(True)
    metrics_plot_path = os.path.join(SESSION_FOLDER, f"metrics_plot_{current_time_str}.png")
    plt.tight_layout(); plt.savefig(metrics_plot_path); plt.close(fig)

    cm = confusion_matrix(all_true, all_preds, labels=range(len(class_names)))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix'); plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    cm_plot_path = os.path.join(SESSION_FOLDER, f"confusion_matrix_{current_time_str}.png")
    plt.tight_layout(); plt.savefig(cm_plot_path); plt.close()
    print("Đã lưu các biểu đồ hiệu suất.")

    # --- BẮT ĐẦU PHẦN XAI / GRAD-CAM ---
    print("\nBắt đầu phân tích XAI với Grad-CAM...")
    
    def get_grad_cam_image(model, original_file_path, target_class_idx):
        try:
            # Lớp mục tiêu cho AST
            target_layers = [model.audio_spectrogram_transformer.encoder.layer[-1].output]
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
            
            waveform = preprocess_audio(original_file_path, CONFIG)
            if waveform is None: return None
            
            spec = waveform_to_mel_spectrogram(waveform, CONFIG)
            
            # Chuẩn bị input tensor đúng như lúc train
            processed_input = feature_extractor(spec, sampling_rate=feature_extractor.sampling_rate, return_tensors="pt")
            input_tensor = processed_input['input_values'].to(device)

            targets = [ClassifierOutputTarget(target_class_idx)]
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets, aug_smooth=True, eigen_smooth=True)[0, :]
            
            # Visualize
            spec_rgb = np.stack([spec]*3, axis=-1)
            visualization = show_cam_on_image(spec_rgb, grayscale_cam, use_rgb=True)
            return visualization
        except Exception as e:
            print(f"Lỗi khi tạo Grad-CAM cho {original_file_path}: {e}")
            return None
    
    # Phân tích các mẫu bị phân loại sai
    error_xai_paths = []
    misclassified_indices = np.where(all_preds != all_true)[0]
    df_errors = pd.DataFrame({
        'true_label': all_true[misclassified_indices],
        'pred_label': all_preds[misclassified_indices],
        'confidence': [all_probs[i][all_preds[i]] for i in misclassified_indices],
        'filepath': np.array(all_original_paths)[misclassified_indices]
    })
    
    for class_idx in range(len(class_names)):
        top_3_errors = df_errors[df_errors['true_label'] == class_idx].nlargest(3, 'confidence')
        for i, row in top_3_errors.iterrows():
            vis = get_grad_cam_image(model, row['filepath'], row['pred_label'])
            if vis is not None:
                path = os.path.join(SESSION_FOLDER, f"error_XAI_{idx_to_class[class_idx]}_as_{idx_to_class[row['pred_label']]}_{i}.png")
                Image.fromarray(vis).save(path)
                info = (f"Lỗi #{i} (Lớp thật: {idx_to_class[class_idx]}): Dự đoán là '{idx_to_class[row['pred_label']]}' (Conf: {row['confidence']:.2f})", path)
                error_xai_paths.append(info)
    
    # --- Tạo Báo Cáo PDF ---
    print("\nBắt đầu tạo báo cáo PDF cuối cùng...")
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Vietnamese', fontName=PDF_FONT_NAME, fontSize=10))
    styles.add(ParagraphStyle(name='Vietnamese_h1', parent=styles['h1'], fontName=PDF_FONT_NAME))
    styles.add(ParagraphStyle(name='Vietnamese_h2', parent=styles['h2'], fontName=PDF_FONT_NAME))
    
    story = [
        Paragraph("BÁO CÁO KẾT QUẢ HUẤN LUYỆN MÔ HÌNH", styles['Vietnamese_h1']),
        Paragraph(f"Thời gian: {current_time_str} | Mô hình: {CONFIG.MODEL_ID}", styles['Vietnamese']), Spacer(1, 12),
        Paragraph(f"Các lớp: {CONFIG.CLASSES_TO_USE}", styles['Vietnamese']), Spacer(1, 12),
        Paragraph("I. Biểu đồ Huấn luyện & Đánh giá", styles['Vietnamese_h2']),
        ReportlabImage(metrics_plot_path, width=450, height=550), PageBreak(),
        Paragraph("II. Báo cáo & Ma trận Nhầm lẫn", styles['Vietnamese_h2']),
        ReportlabImage(cm_plot_path, width=400, height=300), Spacer(1, 12)
    ]
    report_data = classification_report(all_true, all_preds, target_names=class_names, labels=range(len(class_names)), output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_data).transpose().round(2).reset_index()
    table = Table([report_df.columns.tolist()] + report_df.values.tolist())
    table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey), ('GRID', (0,0), (-1,-1), 1, colors.black), ('FONTNAME', (0,0), (-1,-1), PDF_FONT_NAME)]))
    story.append(table)

    if error_xai_paths:
        story.append(PageBreak())
        story.append(Paragraph("III. Phân tích Lỗi sai & XAI (Top 3 mỗi lớp)", styles['Vietnamese_h2']))
        for title, path in error_xai_paths:
            if os.path.exists(path):
                story.extend([Paragraph(title, styles['Vietnamese']), ReportlabImage(path, width=400, height=400), Spacer(1, 12)])
    
    pdf_report_path = os.path.join(SESSION_FOLDER, f"final_report_{current_time_str}.pdf")
    SimpleDocTemplate(pdf_report_path, pagesize=letter).build(story)

    print(f"\nBÁO CÁO PDF HOÀN CHỈNH ĐÃ ĐƯỢC LƯU TẠI: {pdf_report_path}")
    print("\nPIPELINE HOÀN TẤT!")