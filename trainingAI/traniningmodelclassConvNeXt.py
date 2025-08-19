# ======================================================================================
# BLOCK 1: CÀI ĐẶT, IMPORT VÀ CẤU HÌNH NÂNG CAO (Version Final)
# ======================================================================================
print("BLOCK 1: CÀI ĐẶT, IMPORT VÀ CẤU HÌNH NÂNG CAO...")

# Cài đặt các thư viện cần thiết
!pip install -q timm torch torchaudio scikit-learn pandas matplotlib seaborn librosa pydub pytz reportlab grad-cam transformers noisereduce -q

import os
import sys
import random
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
from sklearn.utils.class_weight import compute_class_weight
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
import requests
from itertools import cycle
from PIL import Image
import contextlib
import noisereduce as nr
from torch.cuda.amp import GradScaler, autocast

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

# Sử dụng font hệ thống để đảm bảo ổn định
VIETNAMESE_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
PDF_FONT_NAME = "DejaVuSans"

print("Dang kiem tra va dang ky font tieng Viet cho bao cao PDF...")
if os.path.exists(VIETNAMESE_FONT_PATH):
    try:
        pdfmetrics.registerFont(TTFont(PDF_FONT_NAME, VIETNAMESE_FONT_PATH))
        print(f"Dang ky font '{PDF_FONT_NAME}' tu he thong thanh cong.")
    except Exception as e:
        print(f"Loi khi dang ky font he thong: {e}")
        PDF_FONT_NAME = "Helvetica" # Fallback an toan
        print(f"Khong the dang ky font. Se su dung font mac dinh '{PDF_FONT_NAME}'.")
else:
    PDF_FONT_NAME = "Helvetica" # Fallback an toan
    print(f"Khong tim thay font DejaVu Sans. Se su dung font mac dinh '{PDF_FONT_NAME}'.")


# --- Cấu hình toàn bộ pipeline ---
class CONFIG:
    # 💡 CÔNG TẮC ĐIỀU KHIỂN CHẾ ĐỘ CHẠY 💡
    RUN_PREPROCESSING = False

    # --- CÔNG TẮC KHÔI PHỤC ---
    RESUME_TRAINING = False
    RESUME_SESSION_FOLDER_NAME = ""

    # --- Cấu hình Mô hình & Huấn luyện ---
    MODEL_ID = "convnext_tiny"
    EPOCHS = 50
    LEARNING_RATE = 1e-5
    BATCH_SIZE = 64
    SEED = 42

    # --- Cấu hình Chống Overfitting ---
    EARLY_STOPPING_PATIENCE = 10
    USE_DATA_AUGMENTATION = False

    # --- Cấu hình Dữ liệu & Tiền xử lý ---
    DATA_PATH = "/content/drive/MyDrive/Tai_Lieu_NCKH/dataset_cough"
    PREPROCESSED_DATA_PATH = "Tai_Lieu_NCKH/spectrogram_3channeldata"

    # --- Cấu hình Google Drive & Output ---
    DRIVE_MOUNT_PATH = "/content/drive"
    DRIVE_OUTPUT_PATH = "Tai_Lieu_NCKH/newAiData/ConvNetXt"
    CHECKPOINT_SUBFOLDER = "checkpoints"

    # --- Cấu hình Timezone ---
    TIMEZONE = "Asia/Ho_Chi_Minh"

    # --- Các cấu hình khác ---
    IMAGE_SIZE = 224
    SAMPLE_RATE = 16000
    MAX_LENGTH_SECS = 5
    N_MELS = 224
    N_FFT = 2048
    HOP_LENGTH = 512
    SILENCE_THRESH = -40
    MIN_SILENCE_LEN = 300

# --- Hàm thiết lập Seed ---
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    print(f"Da thiet lap Seed cho toan bo script la: {seed_value}")

set_seed(CONFIG.SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Su dung thiet bi: {device}")

# --- Setup Môi trường ---
drive.mount(CONFIG.DRIVE_MOUNT_PATH)

# --- Tạo các đường dẫn cần thiết ---
BASE_DRIVE_OUTPUT_PATH = os.path.join(CONFIG.DRIVE_MOUNT_PATH, "MyDrive", CONFIG.DRIVE_OUTPUT_PATH)
FULL_PREPROCESSED_PATH = os.path.join(CONFIG.DRIVE_MOUNT_PATH, "MyDrive", CONFIG.PREPROCESSED_DATA_PATH)
os.makedirs(BASE_DRIVE_OUTPUT_PATH, exist_ok=True)
os.makedirs(FULL_PREPROCESSED_PATH, exist_ok=True)

tz = pytz.timezone(CONFIG.TIMEZONE)
def get_vn_time_str():
    return datetime.datetime.now(tz).strftime("%Y-%m-%d_%H-%M-%S")

if not CONFIG.RUN_PREPROCESSING:
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
# BLOCK 2: CÁC HÀM TIỀN XỬ LÝ VÀ TRÍCH XUẤT ĐẶC TRƯNG
# ======================================================================================
print("\nBLOCK 2: KHỞI TẠO CÁC HÀM TIỀN XỬ LÝ...")
def preprocess_audio(file_path):
    try:
        samples, sr = librosa.load(file_path, sr=CONFIG.SAMPLE_RATE)
        samples_denoised = nr.reduce_noise(y=samples, sr=sr)
        int_samples = (samples_denoised * 32767).astype(np.int16)
        sound = AudioSegment(int_samples.tobytes(), frame_rate=sr, sample_width=int_samples.dtype.itemsize, channels=1)
        chunks = split_on_silence(sound, min_silence_len=CONFIG.MIN_SILENCE_LEN, silence_thresh=CONFIG.SILENCE_THRESH)
        processed_sound = sum(chunks, AudioSegment.empty()) if chunks else sound
        final_samples = np.array(processed_sound.get_array_of_samples()).astype(np.float32)
        if len(final_samples) > 0:
            final_samples /= np.iinfo(processed_sound.sample_width * 8).max
        max_samples = CONFIG.SAMPLE_RATE * CONFIG.MAX_LENGTH_SECS
        if len(final_samples) > max_samples: final_samples = final_samples[:max_samples]
        else: final_samples = np.pad(final_samples, (0, max_samples - len(final_samples)), 'constant')
        return final_samples
    except Exception as e:
        print(f"Lỗi xử lý file {file_path}: {e}"); return None

def waveform_to_3_channel_features(waveform):
    mel_spec = librosa.feature.melspectrogram(y=waveform, sr=CONFIG.SAMPLE_RATE, n_fft=CONFIG.N_FFT, hop_length=CONFIG.HOP_LENGTH, n_mels=CONFIG.N_MELS)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    delta1 = librosa.feature.delta(log_mel_spec)
    S = np.abs(librosa.stft(waveform, n_fft=CONFIG.N_FFT, hop_length=CONFIG.HOP_LENGTH))
    contrast = librosa.feature.spectral_contrast(S=S, sr=CONFIG.SAMPLE_RATE)
    contrast_resized = np.array(Image.fromarray(contrast).resize((log_mel_spec.shape[1], log_mel_spec.shape[0]), Image.Resampling.LANCZOS))
    def normalize_channel(channel):
        if channel.max() == channel.min(): return np.zeros_like(channel)
        return (channel - channel.min()) / (channel.max() - channel.min())
    log_mel_spec_norm = normalize_channel(log_mel_spec)
    delta1_norm = normalize_channel(delta1)
    contrast_norm = normalize_channel(contrast_resized)
    three_channel_img = np.stack([log_mel_spec_norm, delta1_norm, contrast_norm], axis=-1)
    return three_channel_img

print("Hoan tat Block 2.")


# ======================================================================================
# --- BLOCK CHÍNH: LỰA CHỌN CHẾ ĐỘ CHẠY (TIỀN XỬ LÝ hoặc HUẤN LUYỆN) ---
# ======================================================================================
if CONFIG.RUN_PREPROCESSING:
    # --- CHẾ ĐỘ 1: TIỀN XỬ LÝ DỮ LIỆU (CHẠY 1 LẦN) ---
    print("\n========================================================================")
    print("                      CHẾ ĐỘ: TIỀN XỬ LÝ DỮ LIỆU")
    print("========================================================================\n")

    print("Đang quét và tạo DataFrame ban đầu...")
    all_data = []
    class_names = sorted([d for d in os.listdir(CONFIG.DATA_PATH) if os.path.isdir(os.path.join(CONFIG.DATA_PATH, d))])
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(CONFIG.DATA_PATH, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.lower().endswith(".wav"):
                file_path = os.path.join(class_dir, file_name)
                participant_id = ""
                if class_name in ['healthy', 'covid', 'asthma']: participant_id = file_name.split('_')[0]
                elif class_name == 'tuberculosis':
                    if '_' in file_name: participant_id = file_name.rsplit('_', 1)[0]
                    else: participant_id = file_name
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

    def process_and_save_features(dataframe, set_name, base_output_dir):
        print(f"\nBắt đầu xử lý tập {set_name}...")
        new_paths = []
        for index, row in tqdm(dataframe.iterrows(), total=len(dataframe), desc=f"Processing {set_name}"):
            original_path, label_idx = row['file_path'], row['label']
            class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(label_idx)]
            output_class_dir = os.path.join(base_output_dir, set_name, class_name)
            os.makedirs(output_class_dir, exist_ok=True)
            filename_without_ext = os.path.splitext(os.path.basename(original_path))[0]
            output_npy_path = os.path.join(output_class_dir, f"{filename_without_ext}.npy")
            
            waveform = preprocess_audio(original_path)
            if waveform is not None and np.any(waveform):
                features_3_channel = waveform_to_3_channel_features(waveform)
                np.save(output_npy_path, features_3_channel)
                new_paths.append(output_npy_path)
            else:
                new_paths.append(None)
        dataframe['spectrogram_path'] = new_paths
        dataframe.dropna(subset=['spectrogram_path'], inplace=True)
        return dataframe

    train_df_processed = process_and_save_features(train_df, 'train', FULL_PREPROCESSED_PATH)
    val_df_processed = process_and_save_features(val_df, 'val', FULL_PREPROCESSED_PATH)
    test_df_processed = process_and_save_features(test_df, 'test', FULL_PREPROCESSED_PATH)

    train_df_processed.to_csv(os.path.join(FULL_PREPROCESSED_PATH, "train.csv"), index=False)
    val_df_processed.to_csv(os.path.join(FULL_PREPROCESSED_PATH, "val.csv"), index=False)
    test_df_processed.to_csv(os.path.join(FULL_PREPROCESSED_PATH, "test.csv"), index=False)

    print("\n--- TIỀN XỬ LÝ HOÀN TẤT! ---")
    print(f"Đã lưu {len(train_df_processed)} spectrograms cho tập train.")
    print(f"Đã lưu {len(val_df_processed)} spectrograms cho tập validation.")
    print(f"Đã lưu {len(test_df_processed)} spectrograms cho tập test.")
    print(f"Thông tin các tập dữ liệu đã được lưu tại các file .csv trong: {FULL_PREPROCESSED_PATH}")
    print("\n💡 HƯỚNG DẪN: Bây giờ hãy đặt CONFIG.RUN_PREPROCESSING = False và chạy lại toàn bộ script để bắt đầu huấn luyện.")

else:
    # --- CHẾ ĐỘ 2: HUẤN LUYỆN VÀ ĐÁNH GIÁ ---
    print("\n========================================================================")
    print("                  CHẾ ĐỘ: HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH")
    print("========================================================================\n")

    # ======================================================================================
    # BLOCK 3: DATASET VÀ DATALOADER
    # ======================================================================================
    print("\nBLOCK 3: KHỞI TẠO DATASET VÀ DATALOADER TỪ DỮ LIỆU ĐÃ TIỀN XỬ LÝ...")

    train_transforms_list = [transforms.Resize((CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE))]
    if CONFIG.USE_DATA_AUGMENTATION:
        print("Data Augmentation da duoc BAT.")
        train_transforms_list.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(10)])
    else: print("Data Augmentation da duoc TAT.")
    train_transforms_list.extend([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    data_transforms = {
        'train': transforms.Compose(train_transforms_list),
        'val': transforms.Compose([
            transforms.Resize((CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
    }

    class CoughDataset(Dataset):
        def __init__(self, file_paths, labels, transform):
            self.file_paths, self.labels, self.transform = file_paths, labels, transform
        def __len__(self): return len(self.file_paths)
        def __getitem__(self, idx):
            spectrogram_path = self.file_paths[idx]
            try:
                features_3_channel = np.load(spectrogram_path)
                spec_img = Image.fromarray((features_3_channel * 255).astype(np.uint8))
                image_tensor = self.transform(spec_img)
                return {"image": image_tensor, "labels": torch.tensor(self.labels[idx], dtype=torch.long)}
            except Exception as e:
                print(f"Lỗi khi tải hoặc xử lý file {spectrogram_path}: {e}")
                return {"image": torch.zeros(3, CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE), "labels": torch.tensor(-1, dtype=torch.long)}

    print("Đang tải thông tin các tập dữ liệu từ file CSV...")
    try:
        train_df = pd.read_csv(os.path.join(FULL_PREPROCESSED_PATH, "train.csv"))
        val_df = pd.read_csv(os.path.join(FULL_PREPROCESSED_PATH, "val.csv"))
        test_df = pd.read_csv(os.path.join(FULL_PREPROCESSED_PATH, "test.csv"))
    except FileNotFoundError:
        sys.exit(f"LỖI: Không tìm thấy file CSV. Hãy đặt CONFIG.RUN_PREPROCESSING = True và chạy lại script để tạo dữ liệu.")

    class_names = sorted(np.unique(train_df['label']).astype(int))
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    idx_to_class = {i: name for name, i in class_to_idx.items()}

    train_files, train_labels = train_df['spectrogram_path'].tolist(), train_df['label'].tolist()
    val_files, val_labels = val_df['spectrogram_path'].tolist(), val_df['label'].tolist()
    test_files, test_labels = test_df['spectrogram_path'].tolist(), test_df['label'].tolist()

    print(f"\nChia du lieu hoan tat (tải từ file):")
    print(f" - Tap Train: {len(train_files)} files.")
    print(f" - Tap Validation: {len(val_files)} files.")
    print(f" - Tap Test: {len(test_files)} files.")

    print("\nDang tinh toan Class Weights...")
    class_weights = compute_class_weight('balanced', classes=np.unique(train_labels), y=train_labels)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    for i, name in idx_to_class.items(): print(f" - Lop '{class_names[i]}' (id {i}): weight = {class_weights[i]:.2f}")

    train_dataset = CoughDataset(train_files, train_labels, data_transforms['train'])
    val_dataset = CoughDataset(val_files, val_labels, data_transforms['val'])
    test_dataset = CoughDataset(test_files, test_labels, data_transforms['val'])
    test_dataset.original_file_paths = test_df['file_path'].tolist()

    train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.BATCH_SIZE, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG.BATCH_SIZE, num_workers=2, pin_memory=True)
    print("\nHoan tat Block 3.")

    # ======================================================================================
    # BLOCK 4: MÔ HÌNH VÀ CÁC TIỆN ÍCH NÂNG CAO
    # ======================================================================================
    print("\nBLOCK 4: KHỞI TẠO MÔ HÌNH VÀ CÁC TIỆN ÍCH...")
    def load_model(num_labels, pretrained=True):
        model = timm.create_model(CONFIG.MODEL_ID, pretrained=pretrained, num_classes=num_labels)
        return model.to(device)

    class EarlyStopping:
        def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth.tar'):
            self.patience, self.verbose, self.delta, self.path = patience, verbose, delta, path
            self.counter, self.best_score, self.early_stop, self.val_loss_min = 0, None, False, np.inf
        def __call__(self, val_loss, model_state):
            score = -val_loss
            if self.best_score is None or score > self.best_score + self.delta:
                if self.verbose: print(f'Validation loss giam ({self.val_loss_min:.6f} --> {val_loss:.6f}). Dang luu model...')
                torch.save(model_state, self.path)
                self.val_loss_min, self.best_score, self.counter = val_loss, score, 0
            else:
                self.counter += 1
                if self.verbose: print(f'EarlyStopping counter: {self.counter} / {self.patience}')
                if self.counter >= self.patience: self.early_stop = True

    def load_checkpoint(model, optimizer, scheduler, folder):
        filepath = os.path.join(folder, "last_checkpoint.pth.tar")
        if os.path.exists(filepath):
            print(f"Tai checkpoint tu {filepath}")
            checkpoint = torch.load(filepath, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            return checkpoint['epoch'] + 1, checkpoint['history']
        return 0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    def save_checkpoint(state, folder):
        torch.save(state, os.path.join(folder, "last_checkpoint.pth.tar"))
    print("Hoan tat Block 4.")

    # ======================================================================================
    # BLOCK 5: VÒNG LẶP HUẤN LUYỆN (TÍCH HỢP AMP)
    # ======================================================================================
    print("\nBLOCK 5: BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN...")
    model = load_model(len(class_names))
    # model = torch.compile(model)
    optimizer = AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    num_training_steps = CONFIG.EPOCHS * len(train_loader)
    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=int(0.1*num_training_steps), num_training_steps=num_training_steps)
    early_stopper = EarlyStopping(patience=CONFIG.EARLY_STOPPING_PATIENCE, verbose=True, path=os.path.join(CHECKPOINT_FOLDER, "best_model.pth.tar"))
    start_epoch, history = load_checkpoint(model, optimizer, lr_scheduler, CHECKPOINT_FOLDER) if CONFIG.RESUME_TRAINING else (0, {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []})

    scaler = GradScaler()
    for epoch in range(start_epoch, CONFIG.EPOCHS):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG.EPOCHS} ---")
        model.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        progress_bar = tqdm(train_loader, desc="Training")
        for batch in progress_bar:
            if -1 in batch['labels']: continue
            images, labels = batch['image'].to(device), batch['labels'].to(device)
            optimizer.zero_grad()
            with autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            predictions = torch.argmax(logits, dim=-1)
            total_loss += loss.item() * images.size(0)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
            if total_samples > 0: progress_bar.set_postfix(loss=total_loss/total_samples, acc=total_correct/total_samples)
        train_loss, train_acc = (total_loss/total_samples, total_correct/total_samples) if total_samples > 0 else (0,0)
        history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)

        model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if -1 in batch['labels']: continue
                images, labels = batch['image'].to(device), batch['labels'].to(device)
                with autocast():
                    logits = model(images)
                    loss = criterion(logits, labels)
                total_loss += loss.item() * images.size(0)
                total_correct += (torch.argmax(logits, dim=-1) == labels).sum().item()
                total_samples += labels.size(0)
        val_loss, val_acc = (total_loss/total_samples, total_correct/total_samples) if total_samples > 0 else (0,0)
        history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)
        print(f"Epoch {epoch + 1} Summary: Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
        save_checkpoint({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': lr_scheduler.state_dict(), 'history': history}, CHECKPOINT_FOLDER)
        early_stopper(val_loss, model.state_dict())
        if early_stopper.early_stop: print("Early stopping triggered!"); break
    print("\nHoan tat qua trinh huan luyen.")

    # ======================================================================================
    # BLOCK 6: PHÂN TÍCH TOÀN DIỆN VÀ BÁO CÁO KẾT QUẢ
    # ======================================================================================
    print("\nBLOCK 6: BẮT ĐẦU ĐÁNH GIÁ, PHÂN TÍCH SÂU VÀ TẠO BÁO CÁO...")
    best_model_path = os.path.join(CHECKPOINT_FOLDER, "best_model.pth.tar")
    if os.path.exists(best_model_path):
        print(f"Tai lai trong so tot nhat tu: {best_model_path}")
        model = load_model(len(class_names), pretrained=False)
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print("Canh bao: Khong tim thay best_model.pth.tar. Su dung mo hinh cuoi cung de danh gia.")
    current_time_str = get_vn_time_str()
    model.eval()
    print("Bat dau thu thap ket qua, xac suat va embeddings tu tap Test...")
    all_preds, all_true, all_probs, all_embeddings = [], [], [], []
    test_filepaths_in_order = test_dataset.original_file_paths
    with torch.no_grad():
        for batch in tqdm(test_loader, "Collecting Test Set Results"):
            if -1 in batch['labels']: continue
            images, labels = batch['image'].to(device), batch['labels'].to(device)
            features = model.forward_features(images)
            embeddings = model.forward_head(features, pre_logits=True)
            logits = model.head(embeddings)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            predictions = torch.argmax(probs, dim=-1)
            all_probs.extend(probs.cpu().numpy()); all_preds.extend(predictions.cpu().numpy()); all_true.extend(labels.cpu().numpy()); all_embeddings.extend(embeddings.cpu().numpy())
    all_true = np.array(all_true); all_preds = np.array(all_preds); all_probs = np.array(all_probs); all_embeddings = np.array(all_embeddings)
    print("\nDang tao va luu cac bieu do hieu suat...")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10)); ax1.plot(history['train_loss'], label='Train Loss'); ax1.plot(history['val_loss'], label='Validation Loss'); ax1.set_title('Training & Validation Loss'); ax1.legend(); ax1.grid(True); ax2.plot(history['train_acc'], label='Train Accuracy'); ax2.plot(history['val_acc'], label='Validation Accuracy'); ax2.set_title('Training & Validation Accuracy'); ax2.legend(); ax2.grid(True)
    metrics_plot_path = os.path.join(SESSION_FOLDER, f"metrics_plot_{current_time_str}.png"); plt.tight_layout(); plt.savefig(metrics_plot_path); plt.close(fig)
    cm = confusion_matrix(all_true, all_preds); plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names); plt.title('Confusion Matrix'); plt.ylabel('True Label'); plt.xlabel('Predicted Label'); cm_plot_path = os.path.join(SESSION_FOLDER, f"confusion_matrix_{current_time_str}.png"); plt.tight_layout(); plt.savefig(cm_plot_path); plt.close()
    y_true_binarized = label_binarize(all_true, classes=range(len(class_names))); fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8)); line_colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple'])
    for i, color in zip(range(len(class_names)), line_colors):
        fpr, tpr, _ = roc_curve(y_true_binarized[:, i], all_probs[:, i]); roc_auc = auc(fpr, tpr); ax1.plot(fpr, tpr, color=color, lw=2, label=f'ROC {idx_to_class[i]} (AUC = {roc_auc:0.2f})')
        precision, recall, _ = precision_recall_curve(y_true_binarized[:, i], all_probs[:, i]); ax2.plot(recall, precision, color=color, lw=2, label=f'PR {idx_to_class[i]}')
    ax1.plot([0, 1], [0, 1], 'k--', lw=2); ax1.set_title('Multi-class ROC Curve'); ax1.set_xlabel('False Positive Rate'); ax1.set_ylabel('True Positive Rate'); ax1.legend(loc="lower right"); ax1.grid(True)
    ax2.set_title('Multi-class Precision-Recall Curve'); ax2.set_xlabel('Recall'); ax2.set_ylabel('Precision'); ax2.legend(loc="best"); ax2.grid(True)
    roc_pr_plot_path = os.path.join(SESSION_FOLDER, f"roc_pr_plot_{current_time_str}.png"); plt.savefig(roc_pr_plot_path); plt.close(fig)
    print("Da luu cac bieu do hieu suat.")

    print("\nBat dau tao bieu do t-SNE..."); tsne = TSNE(n_components=2, verbose=0, perplexity=min(30, len(all_embeddings)-1), n_iter=300, random_state=CONFIG.SEED); tsne_results = tsne.fit_transform(all_embeddings); df_tsne = pd.DataFrame(tsne_results, columns=['tsne-1', 'tsne-2']); df_tsne['label'] = [idx_to_class[i] for i in all_true]
    plt.figure(figsize=(10, 10)); sns.scatterplot(x="tsne-1", y="tsne-2", hue="label", palette=sns.color_palette("hsv", len(class_names)), data=df_tsne, legend="full", alpha=0.8); plt.title("t-SNE Visualization of Cough Embeddings"); plt.grid(True)
    tsne_plot_path = os.path.join(SESSION_FOLDER, f"tsne_plot_{current_time_str}.png"); plt.savefig(tsne_plot_path); plt.close()
    print("Da luu bieu do t-SNE.")

    def get_grad_cam_array(model, file_path, target_class_idx):
        try:
            target_layers = [model.stages[-1].blocks[-1]]
            cam = GradCAM(model=model, target_layers=target_layers, use_cuda=torch.cuda.is_available())
            waveform = preprocess_audio(file_path)
            if waveform is None or not np.any(waveform): return None, None
            features_3_channel = waveform_to_3_channel_features(waveform)
            input_tensor = data_transforms['val'](Image.fromarray((features_3_channel * 255).astype(np.uint8))).unsqueeze(0)
            targets = [ClassifierOutputTarget(target_class_idx)]
            grayscale_cam = cam(input_tensor=input_tensor.to(device), targets=targets)[0, :]
            return grayscale_cam, features_3_channel
        except Exception as e:
            print(f"Loi khi tao Grad-CAM cho {file_path}: {e}")
            return None, None

    def visualize_single_grad_cam(model, file_path, output_path, pred_label_idx):
        grayscale_cam, spec_rgb = get_grad_cam_array(model, file_path, pred_label_idx)
        if grayscale_cam is not None:
            visualization = show_cam_on_image(spec_rgb, grayscale_cam, use_rgb=True)
            Image.fromarray(visualization).save(output_path)

    print("\nBat dau phan tich cac mau bi phan loai sai..."); misclassified_indices = np.where(all_preds != all_true)[0]; error_analysis, error_xai_paths = [], []
    for idx in misclassified_indices: error_analysis.append((all_probs[idx][all_preds[idx]], all_true[idx], all_preds[idx], test_filepaths_in_order[idx]))
    error_analysis.sort(key=lambda x: x[0], reverse=True)
    print(f"\n--- Top 5 Loi sai Tu tin nhat ---")
    for i, (confidence, true_label_idx, pred_label_idx, file_path) in enumerate(error_analysis[:5]):
        error_xai_path = os.path.join(SESSION_FOLDER, f"error_analysis_sample_{i+1}_{current_time_str}.png"); visualize_single_grad_cam(model, file_path, error_xai_path, pred_label_idx)
        error_info = (f"Loi #{i+1}: '{os.path.basename(file_path)}' | That: '{idx_to_class[true_label_idx]}' | Sai: '{idx_to_class[pred_label_idx]}' (Conf: {confidence:.2f})", error_xai_path)
        error_xai_paths.append(error_info); print(error_info[0])

    print("\nBat dau tao Grad-CAM trung binh cho cac du doan dung...")
    avg_cam_paths = []; correct_indices = np.where(all_preds == all_true)[0]; df_correct = pd.DataFrame({'filepath': np.array(test_filepaths_in_order)[correct_indices], 'true_label': all_true[correct_indices], 'confidence': np.max(all_probs[correct_indices], axis=1)})
    for class_idx, class_name in idx_to_class.items():
        print(f" - Dang xu ly lop: {class_name}")
        top_10_df = df_correct[df_correct['true_label'] == class_idx].nlargest(10, 'confidence')
        if top_10_df.empty: print(f"   -> Khong co du doan dung nao cho lop {class_name} trong tap test."); continue
        cam_arrays, background_spec = [], None
        for _, row in top_10_df.iterrows():
            grayscale_cam, spec_rgb = get_grad_cam_array(model, row['filepath'], class_idx)
            if grayscale_cam is not None:
                cam_arrays.append(grayscale_cam)
                if background_spec is None: background_spec = spec_rgb
        if not cam_arrays: print(f"   -> Khong the tao Grad-CAM cho lop {class_name}."); continue
        avg_cam = np.mean(cam_arrays, axis=0); avg_visualization = show_cam_on_image(background_spec, avg_cam, use_rgb=True)
        avg_cam_path = os.path.join(SESSION_FOLDER, f"average_gradcam_{class_name}_{current_time_str}.png"); Image.fromarray(avg_visualization).save(avg_cam_path)
        avg_cam_info = (f"Grad-CAM Trung binh cho lop '{idx_to_class[class_idx]}' (dua tren {len(cam_arrays)} mau tu tin nhat)", avg_cam_path); avg_cam_paths.append(avg_cam_info); print(f"   -> Da luu Grad-CAM trung binh.")

    print("\nBat dau tao bao cao PDF cuoi cung..."); styles = getSampleStyleSheet(); styles.add(ParagraphStyle(name='Vietnamese', fontName=PDF_FONT_NAME, fontSize=10)); styles.add(ParagraphStyle(name='Vietnamese_h1', parent=styles['h1'], fontName=PDF_FONT_NAME)); styles.add(ParagraphStyle(name='Vietnamese_h2', parent=styles['h2'], fontName=PDF_FONT_NAME))
    story = [Paragraph("BÁO CÁO KẾT QUẢ HUẤN LUYỆN MÔ HÌNH", styles['Vietnamese_h1']), Paragraph(f"Thoi gian: {current_time_str} | Mo hinh: {CONFIG.MODEL_ID}", styles['Vietnamese']), Spacer(1, 12),
             Paragraph("I. Bieu do Huan luyen & Danh gia", styles['Vietnamese_h2']), ReportlabImage(metrics_plot_path, width=450, height=550), PageBreak(),
             Paragraph("II. Bao cao & Ma tran Nham lan", styles['Vietnamese_h2']), ReportlabImage(cm_plot_path, width=400, height=300), Spacer(1, 12)]
    report_df = pd.DataFrame(classification_report(all_true, all_preds, target_names=[idx_to_class[i] for i in class_names], output_dict=True, zero_division=0)).transpose().round(2).reset_index(); table = Table([report_df.columns.tolist()] + report_df.values.tolist(), colWidths=[80] + [50]*(len(class_names)+2)); table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.grey), ('GRID', (0,0), (-1,-1), 1, colors.black), ('FONTNAME', (0,0), (-1,-1), PDF_FONT_NAME)]))
    story.extend([table, PageBreak(), Paragraph("III. Phan tich Hieu suat Nang cao", styles['Vietnamese_h2']), ReportlabImage(roc_pr_plot_path, width=500, height=350), Spacer(1, 24),
                  Paragraph("IV. Truc quan hoa Embedding (t-SNE)", styles['Vietnamese_h2']), ReportlabImage(tsne_plot_path, width=500, height=500), PageBreak(),
                  Paragraph("V. Phan tich Loi sai & XAI (Mau du doan SAI)", styles['Vietnamese_h2'])])
    for title, path in error_xai_paths:
        if os.path.exists(path): story.extend([Paragraph(title, styles['Vietnamese']), ReportlabImage(path, width=400, height=400), Spacer(1, 12)])
    story.append(PageBreak()); story.append(Paragraph("VI. Phan tich XAI tren cac mau du doan ĐUNG (Grad-CAM Trung binh)", styles['Vietnamese_h2']))
    for title, path in avg_cam_paths:
        if os.path.exists(path): story.extend([Paragraph(title, styles['Vietnamese']), ReportlabImage(path, width=400, height=400), Spacer(1, 12)])
    pdf_report_path = os.path.join(SESSION_FOLDER, f"final_report_{current_time_str}.pdf"); SimpleDocTemplate(pdf_report_path, pagesize=letter).build(story)
    print(f"\nBÁO CÁO PDF HOÀN CHỈNH ĐÃ ĐUOC LUU TAI: {pdf_report_path}"); print("\nPIPELINE HOÀN TẤT!")
