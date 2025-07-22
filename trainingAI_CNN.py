# ===================================================================
# BLOCK 1: CÀI ĐẶT, KẾT NỐI VÀ IMPORT
# ===================================================================

# Cài đặt các thư viện cần thiết
!pip install librosa soundfile numpy tqdm audiomentations scikit-learn pytorch-grad-cam psutil fpdf2 seaborn -q

# Import các thư viện
import os
import re
import shutil
import numpy as np
import librosa
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import glob
import time
import psutil
from datetime import datetime
import pytz

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
from torchvision.models import ResNet18_Weights

from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from fpdf import FPDF

from pytorch_grad_cam import GradCam
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Kết nối với Google Drive
from google.colab import drive
drive.mount('/content/drive')


# ===================================================================
# BLOCK 2: CẤU HÌNH TẬP TRUNG (CENTRALIZED CONFIGURATION)
# ===================================================================

# --- Đường dẫn ---
DRIVE_PATH = "/content/drive/MyDrive/"
# THƯ MỤC INPUT: Chứa dữ liệu âm thanh thô (.wav, .mp3)
INPUT_BASE_DIR = os.path.join(DRIVE_PATH, "du_lieu_goc")
# THƯ MỤC OUTPUT GỐC: Nơi script sẽ tạo các thư mục output được đánh dấu thời gian
BASE_OUTPUT_DRIVE_DIR = os.path.join(DRIVE_PATH, "KET_QUA_NGHIEN_CUU")

# --- Hyperparameters cho Model và Huấn luyện ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 5
MIN_DELTA = 0.001

# --- Cấu hình Xử lý Âm thanh ---
SAMPLE_RATE = 16000
DURATION_SECONDS = 5
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

# --- Cấu hình Phân chia Dữ liệu ---
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.15, 0.15
RANDOM_SEED = 42

# --- Cấu hình Tăng cường Dữ liệu ---
augmenter = Compose([
    AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
    TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
    PitchShift(min_semitones=-4, max_semitones=4, p=0.5)
])

# --- Định nghĩa các Lớp ---
CLASSES_TO_PROCESS = {
    "COUGH/healthy": {"is_minority": False}, "COUGH/covid": {"is_minority": True},
    "BREATHING/healthy": {"is_minority": False}, "BREATHING/covid": {"is_minority": True},
    "BREATHING/asthma": {"is_minority": True}, "BREATHING/pneumonia": {"is_minority": True},
    "BREATHING/copd": {"is_minority": False}, "BREATHING/lrit": {"is_minority": True},
}
CLASS_NAMES = sorted(list(set([k.split('/')[1] for k in CLASSES_TO_PROCESS.keys()])))
NUM_CLASSES = len(CLASS_NAMES)


# ===================================================================
# BLOCK 3: CÁC LỚP VÀ HÀM TIỆN ÍCH
# ===================================================================

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, model_path='checkpoint.pth'):
        self.patience, self.min_delta, self.model_path = patience, min_delta, model_path
        self.counter, self.best_score, self.early_stop, self.val_loss_min = 0, None, False, np.Inf
    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
    def save_checkpoint(self, val_loss, model):
        print(f'Validation loss giảm ({self.val_loss_min:.6f} --> {val_loss:.6f}). Đang lưu model...')
        torch.save(model.state_dict(), self.model_path)
        self.val_loss_min = val_loss

class CoughSpectrogramDataset(Dataset):
    def __init__(self, data_dir, class_names):
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.file_paths, self.labels = [], []
        for class_name in class_names:
            paths = glob.glob(os.path.join(data_dir, "**", class_name, "*.npy"), recursive=True)
            self.file_paths.extend(paths)
            self.labels.extend([self.class_to_idx[class_name]] * len(paths))
    def __len__(self): return len(self.file_paths)
    def __getitem__(self, idx):
        file_path, label = self.file_paths[idx], self.labels[idx]
        spectrogram = np.load(file_path)
        spectrogram_tensor = torch.from_numpy(spectrogram).float().unsqueeze(0).repeat(3, 1, 1)
        return spectrogram_tensor, label


# ===================================================================
# BLOCK 4: LỚP PIPELINE CHÍNH
# ===================================================================

class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        self.model = None
        self._setup_paths_and_timestamp()
        
    def _setup_paths_and_timestamp(self):
        tz_vietnam = pytz.timezone('Asia/Ho_Chi_Minh')
        self.timestamp = datetime.now(tz_vietnam).strftime('%Y-%m-%d_%H-%M-%S')
        self.output_dir = os.path.join(self.config['BASE_OUTPUT_DRIVE_DIR'], f"output_{self.timestamp}")
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Tất cả output sẽ được lưu tại: {self.output_dir}")
        self.config['FINAL_DATASET_DIR'] = os.path.join(self.output_dir, "final_dataset")
        self.config['MODEL_SAVE_PATH'] = os.path.join(self.output_dir, f"model_{self.timestamp}.pth")

    def run(self):
        data_prepared = self._preprocess_and_split_data()
        if data_prepared:
            self._train_model()
            self._evaluate_and_report()
        print(f"\n🎉 TOÀN BỘ PIPELINE HOÀN TẤT! KẾT QUẢ TẠI: {self.output_dir}")

    def _get_patient_id(self, filename):
        match = re.match(r"([a-zA-Z0-9\-]+)_", filename)
        if match: return match.group(1)
        return os.path.splitext(filename)[0]

    def _preprocess_and_split_data(self):
        print("\n===== QUY TRÌNH 1: XỬ LÝ VÀ CHIA DỮ LIỆU =====")
        cfg = self.config
        patient_files, all_audio_files = defaultdict(list), []
        for root, _, files in os.walk(cfg['INPUT_BASE_DIR']):
            for file in files:
                if file.endswith(('.wav', '.mp3', '.flac')):
                    full_path = os.path.join(root, file)
                    all_audio_files.append(full_path)
                    patient_id = self._get_patient_id(file)
                    patient_files[patient_id].append(full_path)
        
        if not patient_files: 
            print("LỖI: Không tìm thấy file âm thanh trong thư mục input."); return False
        
        all_patient_ids = list(patient_files.keys())
        np.random.seed(cfg['RANDOM_SEED']); np.random.shuffle(all_patient_ids)
        train_idx = int(len(all_patient_ids) * cfg['TRAIN_RATIO'])
        val_idx = train_idx + int(len(all_patient_ids) * cfg['VAL_RATIO'])
        train_ids, val_ids, test_ids = all_patient_ids[:train_idx], all_patient_ids[train_idx:val_idx], all_patient_ids[val_idx:]
        
        patient_to_split_map = {pid: 'train' for pid in train_ids}
        patient_to_split_map.update({pid: 'val' for pid in val_ids})
        patient_to_split_map.update({pid: 'test' for pid in test_ids})

        for source_path in tqdm(all_audio_files, desc="Đang xử lý và chia file"):
            try:
                patient_id = self._get_patient_id(os.path.basename(source_path))
                split_name = patient_to_split_map[patient_id]
                
                signal, _ = librosa.load(source_path, sr=cfg['SAMPLE_RATE'], mono=True)
                target_samples = cfg['DURATION_SECONDS'] * cfg['SAMPLE_RATE']
                if len(signal) > target_samples: signal = signal[:target_samples]
                else: signal = np.pad(signal, (0, target_samples - len(signal)), 'constant')
                
                class_path_part = os.path.relpath(os.path.dirname(source_path), cfg['INPUT_BASE_DIR'])
                if cfg['CLASSES_TO_PROCESS'].get(class_path_part, {}).get("is_minority", False):
                    signal = cfg['augmenter'](samples=signal, sample_rate=cfg['SAMPLE_RATE'])
                
                mel_spec = librosa.feature.melspectrogram(y=signal, sr=cfg['SAMPLE_RATE'], n_fft=cfg['N_FFT'], hop_length=cfg['HOP_LENGTH'], n_mels=cfg['N_MELS'])
                db_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
                
                relative_path = os.path.relpath(source_path, cfg['INPUT_BASE_DIR'])
                destination_path = os.path.join(cfg['FINAL_DATASET_DIR'], split_name, os.path.splitext(relative_path)[0] + ".npy")
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                np.save(destination_path, db_mel_spec)
            except Exception as e:
                print(f"\nLỗi khi xử lý file {source_path}: {e}")
        
        print("\n===== HOÀN TẤT QUY TRÌNH 1 ====="); return True

    def _train_model(self):
        print("\n===== QUY TRÌNH 2: HUẤN LUYỆN MODEL =====")
        cfg = self.config
        train_dataset = CoughSpectrogramDataset(os.path.join(cfg['FINAL_DATASET_DIR'], "train"), cfg['CLASS_NAMES'])
        val_dataset = CoughSpectrogramDataset(os.path.join(cfg['FINAL_DATASET_DIR'], "val"), cfg['CLASS_NAMES'])
        train_loader = DataLoader(train_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg['BATCH_SIZE'], shuffle=False, num_workers=2, pin_memory=True)

        class_weights = torch.tensor(compute_class_weight('balanced', classes=np.arange(cfg['NUM_CLASSES']), y=train_dataset.labels), dtype=torch.float).to(cfg['DEVICE'])
        
        self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.model.fc = nn.Linear(self.model.fc.in_features, cfg['NUM_CLASSES'])
        self.model = self.model.to(cfg['DEVICE'])
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.Adam(self.model.parameters(), lr=cfg['LEARNING_RATE'])
        early_stopping = EarlyStopping(patience=cfg['PATIENCE'], model_path=cfg['MODEL_SAVE_PATH'])

        for epoch in range(cfg['NUM_EPOCHS']):
            self.model.train()
            train_loss = 0.0
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg['NUM_EPOCHS']}"):
                inputs, labels = inputs.to(cfg['DEVICE'], non_blocking=True), labels.to(cfg['DEVICE'], non_blocking=True)
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self.model.eval()
            val_loss, val_corrects = 0.0, 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(cfg['DEVICE'], non_blocking=True), labels.to(cfg['DEVICE'], non_blocking=True)
                    outputs = self.model(inputs)
                    val_loss += criterion(outputs, labels).item()
                    _, preds = torch.max(outputs, 1)
                    val_corrects += torch.sum(preds == labels.data)
            
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_corrects.double() / len(val_dataset) * 100

            self.history['train_loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            self.history['val_acc'].append(val_acc.item())
            
            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            early_stopping(avg_val_loss, self.model)
            if early_stopping.early_stop: print("Dừng sớm do không có cải thiện!"); break
        
        print("\n===== HOÀN TẤT HUẤN LUYỆN =====")

    def _evaluate_and_report(self):
        print("\n===== QUY TRÌNH 3: ĐÁNH GIÁ & TẠO BÁO CÁO =====")
        model = self._load_best_model()
        val_dataset = CoughSpectrogramDataset(os.path.join(self.config['FINAL_DATASET_DIR'], 'val'), self.config['CLASS_NAMES'])
        val_loader = DataLoader(val_dataset, batch_size=self.config['BATCH_SIZE'], shuffle=False, num_workers=2)
        
        y_true, y_pred, y_paths = self._get_predictions(model, val_loader)
        
        if len(y_true) == 0: print("Không có dữ liệu kiểm định để đánh giá."); return

        report_dict = classification_report(y_true, y_pred, target_names=self.config['CLASS_NAMES'], output_dict=True, zero_division=0)
        
        self._generate_plots(y_true, y_pred)
        self._generate_gradcam_visuals(model, val_dataset, y_true, y_pred)
        self._generate_pdf_report(report_dict)

    def _load_best_model(self):
        print(f"Đang tải model tốt nhất từ: {self.config['MODEL_SAVE_PATH']}")
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, self.config['NUM_CLASSES'])
        model.load_state_dict(torch.load(self.config['MODEL_SAVE_PATH']))
        model = model.to(self.config['DEVICE'])
        model.eval()
        return model

    def _get_predictions(self, model, loader):
        model.eval()
        all_labels, all_preds, all_paths = [], [], []
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(loader):
                inputs = inputs.to(self.config['DEVICE'])
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                start_idx = i * loader.batch_size
                end_idx = start_idx + len(labels)
                all_paths.extend(loader.dataset.file_paths[start_idx:end_idx])
        return np.array(all_labels), np.array(all_preds), all_paths
    
    def _generate_plots(self, y_true, y_pred):
        print("Đang tạo các biểu đồ...")
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1); plt.plot(self.history['train_loss'], label='Train Loss'); plt.plot(self.history['val_loss'], label='Val Loss'); plt.title('Training & Validation Loss'); plt.legend()
        plt.subplot(1, 2, 2); plt.plot(self.history['val_acc'], label='Val Accuracy'); plt.title('Validation Accuracy'); plt.legend()
        curves_path = os.path.join(self.output_dir, f"curves_{self.timestamp}.png"); plt.savefig(curves_path); plt.close()
        
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.config['CLASS_NAMES'], yticklabels=self.config['CLASS_NAMES']); plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
        cm_path = os.path.join(self.output_dir, f"confusion_matrix_{self.timestamp}.png"); plt.savefig(cm_path); plt.close()
        print(f"Đã lưu biểu đồ và ma trận nhầm lẫn.")

    def _generate_gradcam_visuals(self, model, dataset, y_true, y_pred):
        print("Đang tạo ảnh Grad-CAM chi tiết...")
        target_layer = model.layer4[-1]
        cam = GradCam(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())

        for class_idx, class_name in enumerate(self.config['CLASS_NAMES']):
            correct_indices = np.where((y_true == class_idx) & (y_pred == class_idx))[0]
            incorrect_indices = np.where((y_true == class_idx) & (y_pred != class_idx))[0]
            
            indices_to_viz = {}
            if len(correct_indices) > 0: indices_to_viz['correct'] = correct_indices[0]
            if len(incorrect_indices) > 0: indices_to_viz['incorrect'] = incorrect_indices[0]

            for viz_type, idx in indices_to_viz.items():
                input_tensor, _ = dataset[idx]
                rgb_img = input_tensor.permute(1, 2, 0).numpy()
                rgb_img = (rgb_img - np.min(rgb_img)) / (np.max(rgb_img) - np.min(rgb_img))
                
                targets = [ClassifierOutputTarget(y_pred[idx])]
                grayscale_cam = cam(input_tensor=input_tensor.unsqueeze(0), targets=targets)[0, :]
                visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
                
                fig, ax = plt.subplots(1, 1); ax.imshow(visualization); ax.axis('off')
                ax.set_title(f"Lop: {class_name} ({viz_type})\nThuc te: {class_name}, Du doan: {self.config['CLASS_NAMES'][y_pred[idx]]}")
                
                save_path = os.path.join(self.output_dir, f"gradcam_{class_name}_{viz_type}_{self.timestamp}.png")
                fig.savefig(save_path); plt.close(fig)
        print("Đã tạo xong ảnh Grad-CAM.")

    def _generate_pdf_report(self, report):
        print("Đang tạo báo cáo PDF...")
        pdf = FPDF()
        pdf.add_page()
        # Thêm font hỗ trợ Unicode (cần có file font .ttf)
        # Tải font từ Google Fonts: https://fonts.google.com/specimen/Roboto
        try:
             # Sửa lại đường dẫn này cho đúng với vị trí file của bạn trên Drive
            font_path = os.path.join(self.config['DRIVE_PATH'], 'Roboto-Regular.ttf') 
            pdf.add_font('Roboto', '', font_path, uni=True)
            pdf.set_font('Roboto', 'B', 16)
        except RuntimeError:
            print("Cảnh báo: Không tìm thấy font Roboto. PDF sẽ không hiển thị tiếng Việt có dấu.")
            pdf.set_font("Arial", 'B', 16)
        
        pdf.cell(0, 10, f"Bao cao HLV Model - {self.timestamp}", ln=True, align='C')
        
        pdf.set_font(pdf.font_family, 'B', 14); pdf.cell(0, 10, "1. Hieu suat qua cac Epoch", ln=True); pdf.ln(5)
        pdf.image(os.path.join(self.output_dir, f"curves_{self.timestamp}.png"), w=190)
        
        pdf.add_page()
        pdf.set_font(pdf.font_family, 'B', 14); pdf.cell(0, 10, "2. Bang ket qua chi tiet", ln=True); pdf.ln(5)
        pdf.set_font(pdf.font_family, 'B', 10); pdf.cell(40, 7, 'Class', 1); pdf.cell(25, 7, 'Precision', 1); pdf.cell(25, 7, 'Recall', 1); pdf.cell(25, 7, 'F1-Score', 1); pdf.cell(25, 7, 'Support', 1); pdf.ln()
        
        pdf.set_font(pdf.font_family, '', 10)
        for class_name, metrics in report.items():
            if isinstance(metrics, dict):
                pdf.cell(40, 7, class_name, 1); pdf.cell(25, 7, f"{metrics.get('precision', 0):.2f}", 1); pdf.cell(25, 7, f"{metrics.get('recall', 0):.2f}", 1); pdf.cell(25, 7, f"{metrics.get('f1-score', 0):.2f}", 1); pdf.cell(25, 7, str(metrics.get('support', 0)), 1); pdf.ln()

        report_path = os.path.join(self.output_dir, f"report_{self.timestamp}.pdf"); pdf.output(report_path)
        print(f"Đã tạo báo cáo PDF.")

# ===================================================================
# BLOCK 5: THỰC THI PIPELINE
# ===================================================================

# Định nghĩa toàn bộ cấu hình
config = {
    "DRIVE_PATH": DRIVE_PATH, "INPUT_BASE_DIR": INPUT_BASE_DIR, "BASE_OUTPUT_DRIVE_DIR": BASE_OUTPUT_DRIVE_DIR,
    "DEVICE": DEVICE, "NUM_EPOCHS": NUM_EPOCHS, "BATCH_SIZE": BATCH_SIZE, 
    "LEARNING_RATE": LEARNING_RATE, "PATIENCE": PATIENCE, "MIN_DELTA": MIN_DELTA,
    "SAMPLE_RATE": SAMPLE_RATE, "DURATION_SECONDS": DURATION_SECONDS, "N_FFT": N_FFT, 
    "HOP_LENGTH": HOP_LENGTH, "N_MELS": N_MELS,
    "TRAIN_RATIO": TRAIN_RATIO, "VAL_RATIO": VAL_RATIO, "TEST_RATIO": TEST_RATIO, "RANDOM_SEED": RANDOM_SEED,
    "augmenter": augmenter, "CLASSES_TO_PROCESS": CLASSES_TO_PROCESS,
    "CLASS_NAMES": CLASS_NAMES, "NUM_CLASSES": NUM_CLASSES
}

# Khởi tạo và chạy pipeline
pipeline = TrainingPipeline(config)
pipeline.run()