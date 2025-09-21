import numpy as np
import soundfile as sf

# --- Các hàm thay thế cho Librosa ---

def to_mono(y):
    """Chuyển đổi âm thanh stereo sang mono bằng cách lấy trung bình."""
    if y.ndim > 1 and y.shape[1] > 1:
        y = np.mean(y, axis=1)
    return y

def normalize(y):
    """Chuẩn hóa audio về khoảng [-1, 1]."""
    max_val = np.max(np.abs(y))
    if max_val > 0:
        return y / max_val
    return y

def trim(y, top_db=40, frame_length=2048, hop_length=512):
    """Loại bỏ khoảng lặng ở đầu và cuối audio."""
    # Tính năng lượng RMS trên từng khung
    rms = np.array([
        np.sqrt(np.mean(y[i:i+frame_length]**2))
        for i in range(0, len(y) - frame_length, hop_length)
    ])
    
    # Chuyển sang dB và chuẩn hóa
    db = 20 * np.log10(rms + 1e-7)
    db_max = np.max(db)
    
    # Tìm các khung có âm lượng lớn hơn ngưỡng
    non_silent_frames = np.where(db > (db_max - top_db))[0]
    
    if len(non_silent_frames) > 0:
        start_frame = non_silent_frames[0]
        end_frame = non_silent_frames[-1]
        
        start_sample = start_frame * hop_length
        end_sample = (end_frame + 1) * hop_length
        return y[start_sample:end_sample]
        
    return y # Trả về mảng gốc nếu không tìm thấy âm thanh

def melspectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128):
    """Tạo Mel Spectrogram từ dữ liệu audio."""
    # 1. Short-Time Fourier Transform (STFT)
    stft_matrix = _stft(y, n_fft, hop_length)
    spectrogram = np.abs(stft_matrix)**2
    
    # 2. Tạo Mel filterbank
    mel_basis = _mel_filterbank(sr, n_fft, n_mels)
    
    # 3. Áp dụng filterbank
    mel_spectrogram = np.dot(mel_basis, spectrogram)
    
    return mel_spectrogram

def power_to_db(S, top_db=80.0):
    """Chuyển đổi power spectrogram sang thang đo decibel (dB)."""
    log_spec = 10.0 * np.log10(np.maximum(1e-10, S))
    max_val = np.max(log_spec)
    
    # Cắt các giá trị quá nhỏ
    log_spec = np.maximum(log_spec, max_val - top_db)
    
    return log_spec

# --- Các hàm helper cho Melspectrogram ---

def _stft(y, n_fft, hop_length):
    """Helper: Tính STFT bằng NumPy."""
    frames = np.lib.stride_tricks.as_strided(
        y,
        shape=(len(y) // hop_length, n_fft),
        strides=(y.strides[0] * hop_length, y.strides[0])
    )
    window = np.hanning(n_fft)
    framed_windowed = frames * window
    return np.fft.rfft(framed_windowed, n=n_fft, axis=1).T

def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def _mel_to_hz(mel):
    return 700.0 * (10.0**(mel / 2595.0) - 1.0)

def _mel_filterbank(sr, n_fft, n_mels):
    """Helper: Tạo một Mel filterbank."""
    low_freq_mel = 0
    high_freq_mel = _hz_to_mel(sr / 2)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bin_points = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    
    filters = np.zeros((n_mels, int(n_fft / 2 + 1)))
    
    for i in range(1, n_mels + 1):
        f_minus = bin_points[i - 1]
        f_center = bin_points[i]
        f_plus = bin_points[i + 1]
        
        for j in range(f_minus, f_center):
            filters[i - 1, j] = (j - f_minus) / (f_center - f_minus)
        for j in range(f_center, f_plus):
            filters[i - 1, j] = (f_plus - j) / (f_plus - f_center)
            
    return filters