################################################################################
# SCRIPT 2: HUẤN LUYỆN VÀ ĐÁNH GIÁ TỪ ĐẶC TRƯNG ĐÃ LƯU
################################################################################

# --- BƯỚC 1: CÀI ĐẶT VÀ IMPORT THƯ VIỆN ---
print("STEP 1: Cài đặt và import thư viện...")
# Lưu ý: Cần chạy, sau đó RESTART RUNTIME để đảm bảo XGBoost phiên bản mới được tải
!pip install -q pandas scikit-learn matplotlib seaborn pytz xgboost tqdm shap openpyxl

from google.colab import drive
import os
import pandas as pd
import numpy as np
import datetime
import pytz
import joblib
import xgboost as xgb
import shap
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

print("STEP 1: Hoàn tất.")

# --- BƯỚC 2: KẾT NỐI GOOGLE DRIVE ---
print("\nSTEP 2: Kết nối với Google Drive...")
drive.mount('/content/drive')
print("-> Kết nối thành công.")

# --- ⚙️ BƯỚC 3: CẤU HÌNH ---
print("\nSTEP 3: Cấu hình...")

# Thư mục cố định trên Drive nơi bạn đã lưu các đặc trưng
FEATURE_STORAGE_FOLDER = '/content/drive/MyDrive/Tai_Lieu_NCKH/newdataset/feature_hybird/'
# Thư mục để lưu kết quả của lần chạy này
OUTPUT_FOLDER = '/content/drive/MyDrive/Tai_Lieu_NCKH/newAiData/'

# Bạn có thể thay đổi các tham số XGBoost ở đây để thử nghiệm
XGB_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}
EARLY_STOPPING_ROUNDS = 20
# ------------------------------------
print("-> Cấu hình hoàn tất.")

vn_timezone = pytz.timezone('Asia/Ho_Chi_Minh')
timestamp = datetime.datetime.now(vn_timezone).strftime('%Y%m%d_%H%M%S')
FINAL_OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, f'output_{timestamp}')
os.makedirs(FINAL_OUTPUT_PATH, exist_ok=True)
print(f"Tất cả kết quả sẽ được lưu tại: {FINAL_OUTPUT_PATH}")

# --- BƯỚC 4: TẢI DỮ LIỆU ĐẶC TRƯNG ĐÃ XỬ LÝ ---
print("\nSTEP 4: Tải dữ liệu đặc trưng đã xử lý...")
try:
    X_train = np.load(os.path.join(FEATURE_STORAGE_FOLDER, 'train_features.npy'))
    y_train = np.load(os.path.join(FEATURE_STORAGE_FOLDER, 'train_labels.npy'))
    X_val = np.load(os.path.join(FEATURE_STORAGE_FOLDER, 'val_features.npy'))
    y_val = np.load(os.path.join(FEATURE_STORAGE_FOLDER, 'val_labels.npy'))
    X_test = np.load(os.path.join(FEATURE_STORAGE_FOLDER, 'test_features.npy'))
    y_test = np.load(os.path.join(FEATURE_STORAGE_FOLDER, 'test_labels.npy'))
    print(f"-> Tải thành công {len(X_train)} mẫu huấn luyện.")
except FileNotFoundError:
    raise FileNotFoundError("Lỗi: Không tìm thấy file đặc trưng. Vui lòng đảm bảo bạn đã chạy Script 1 thành công trước.")


# --- BƯỚC 5: HUẤN LUYỆN MÔ HÌNH ---
print("\nSTEP 5: Huấn luyện mô hình XGBoost...")
classifier = xgb.XGBClassifier(**XGB_PARAMS)
classifier.fit(X_train, y_train,
             eval_set=[(X_val, y_val)],
             early_stopping_rounds=EARLY_STOPPING_ROUNDS,
             verbose=False)
model_path = os.path.join(FINAL_OUTPUT_PATH, f'{timestamp}_xgboost_model.joblib')
joblib.dump(classifier, model_path)
print(f"-> Đã lưu mô hình XGBoost tại: {model_path}")

# --- BƯỚC 6: ĐÁNH GIÁ VÀ GIẢI THÍCH MÔ HÌNH ---
print("\nSTEP 6: Đánh giá và Giải thích mô hình...")
y_pred = classifier.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Độ chính xác cuối cùng trên tập thử nghiệm: {accuracy * 100:.2f}%")

report = classification_report(y_test, y_pred, target_names=['Noise', 'Cough/Breathing'])
print("\nBáo cáo Phân loại:\n", report)
report_path = os.path.join(FINAL_OUTPUT_PATH, f'{timestamp}_report.txt')
with open(report_path, 'w') as f:
    f.write(f"Cấu hình XGBoost:\n{str(XGB_PARAMS)}\n")
    f.write("-" * 50 + "\n")
    f.write(f"Độ chính xác: {accuracy * 100:.2f}%\n\n{report}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Noise', 'Cough/Breathing'], yticklabels=['Noise', 'Cough/Breathing'])
plt.title('Ma trận nhầm lẫn')
plt.savefig(os.path.join(FINAL_OUTPUT_PATH, f'{timestamp}_confusion_matrix.png'))
plt.show()

print("\n--- Bắt đầu tính toán và tạo biểu đồ SHAP ---")
explainer = shap.TreeExplainer(classifier)
# Đối với XGBoost, shap_values sẽ trả về giá trị cho lớp dương tính (lớp 1)
shap_values = explainer.shap_values(X_test) 

# Lấy baseline (expected_value) cho lớp dương tính nếu nó trả về một mảng
base_value = explainer.expected_value
if isinstance(base_value, np.ndarray):
    base_value = base_value[1] # Giả định giá trị thứ 2 là cho lớp 1

# Tạo một list tên đặc trưng giả để biểu đồ dễ nhìn hơn
feature_names = [f'Feature_{i}' for i in range(X_test.shape[1])]

plt.figure()
# Thêm feature_names để hiển thị tên trên biểu đồ
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=30, feature_names=feature_names)
plt.title('SHAP - Mức độ quan trọng của các đặc trưng (Toàn cục)')
plt.tight_layout()
plt.savefig(os.path.join(FINAL_OUTPUT_PATH, f'{timestamp}_shap_summary_bar.png'))
plt.show()

# --- SỬA LỖI LOGIC QUAN TRỌNG ---
# Tạo force_plot cho TOÀN BỘ tập test, không chỉ cho mẫu đầu tiên
print("\nĐang tạo biểu đồ SHAP Force Plot cho toàn bộ tập test...")
shap.initjs()
# Bỏ chỉ số [0,:] để lấy toàn bộ dữ liệu
force_plot = shap.force_plot(base_value, shap_values, X_test, feature_names=feature_names, show=False)
shap.save_html(os.path.join(FINAL_OUTPUT_PATH, f'{timestamp}_shap_force_plot_ALL.html'), force_plot)
print(f"Đã lưu Force Plot cho toàn bộ tập test dưới dạng file HTML.")
# Hiển thị plot cho mẫu đầu tiên để xem trước
display(shap.force_plot(base_value, shap_values[0,:], X_test[0,:], feature_names=feature_names))


print(f"\n--- Pipeline đã hoàn thành! Mọi kết quả đã được lưu tại {FINAL_OUTPUT_PATH} ---")