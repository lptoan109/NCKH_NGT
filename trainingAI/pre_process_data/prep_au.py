import tensorflow as tf
import numpy as np

# ===============================
# IMPROVED PREPROCESSING FOR COUGH SPECTROGRAMS
# ===============================

def spec_augment(spectrogram, num_freq_masks=2, num_time_masks=2,
                 freq_mask_param=27, time_mask_param=100):
    """
    SpecAugment: Random frequency and time masking.
    
    Args:
        spectrogram: 2D or 3D tensor (freq, time) or (freq, time, channels)
        num_freq_masks: Number of frequency masks
        num_time_masks: Number of time masks
        freq_mask_param: Maximum size of frequency mask
        time_mask_param: Maximum size of time mask
    """
    spec = tf.identity(spectrogram)
    freq_dim = tf.shape(spec)[0]
    time_dim = tf.shape(spec)[1]
    
    # Frequency masking
    for _ in range(num_freq_masks):
        # Random mask size (up to freq_mask_param)
        f = tf.minimum(
            tf.random.uniform([], 0, freq_mask_param, dtype=tf.int32),
            freq_dim
        )
        # Random starting position
        f_max = tf.maximum(freq_dim - f, 1)
        f0 = tf.random.uniform([], 0, f_max, dtype=tf.int32)
        
        # Create mask indices
        indices = tf.range(freq_dim)
        mask = tf.cast(
            tf.logical_or(indices < f0, indices >= f0 + f),
            spec.dtype
        )
        # Reshape mask for broadcasting
        if len(spec.shape) == 3:
            mask = tf.reshape(mask, [-1, 1, 1])
        else:
            mask = tf.reshape(mask, [-1, 1])
        
        spec = spec * mask
    
    # Time masking
    for _ in range(num_time_masks):
        t = tf.minimum(
            tf.random.uniform([], 0, time_mask_param, dtype=tf.int32),
            time_dim
        )
        t_max = tf.maximum(time_dim - t, 1)
        t0 = tf.random.uniform([], 0, t_max, dtype=tf.int32)
        
        indices = tf.range(time_dim)
        mask = tf.cast(
            tf.logical_or(indices < t0, indices >= t0 + t),
            spec.dtype
        )
        if len(spec.shape) == 3:
            mask = tf.reshape(mask, [1, -1, 1])
        else:
            mask = tf.reshape(mask, [1, -1])
        
        spec = spec * mask
    
    return spec


def add_gaussian_noise(data, noise_level=0.05):
    """Add Gaussian noise with probability 0.5"""
    if tf.random.uniform([]) > 0.5:
        noise = tf.random.normal(shape=tf.shape(data), mean=0.0, stddev=noise_level)
        data = data + noise
    return data


def normalize_spectrogram(data):
    """Normalize to [-1, 1] range with epsilon for numerical stability"""
    epsilon = 1e-8
    data_min = tf.reduce_min(data)
    data_max = tf.reduce_max(data)
    
    # Normalize to [0, 1]
    data = (data - data_min) / (data_max - data_min + epsilon)
    # Scale to [-1, 1]
    data = (data - 0.5) * 2.0
    
    return data


def preprocess_cough_spectrogram(data, image_size=224, is_training=True,
                                 use_spec_augment=True, noise_level=0.05):
    """
    Preprocessing pipeline for cough spectrograms.
    
    Args:
        data: Input spectrogram (can be 2D or 3D)
        image_size: Target size for resizing
        is_training: Whether to apply augmentations
        use_spec_augment: Whether to use SpecAugment (recommended for audio)
        noise_level: Standard deviation for Gaussian noise
    
    Returns:
        Preprocessed spectrogram tensor of shape (image_size, image_size, 3)
    """
    # Ensure data is float32
    data = tf.cast(data, tf.float32)
    
    # Normalize first (before augmentation for consistency)
    data = normalize_spectrogram(data)
    
    # Convert to 3-channel if needed (before augmentation)
    if len(data.shape) == 2:
        # Apply SpecAugment on 2D spectrogram if training
        if is_training and use_spec_augment:
            data = spec_augment(data, num_freq_masks=2, num_time_masks=2,
                              freq_mask_param=27, time_mask_param=100)
        # Then convert to 3-channel
        data = tf.stack([data, data, data], axis=-1)
    else:
        # If already 3-channel, apply SpecAugment to each channel
        if is_training and use_spec_augment:
            data = spec_augment(data)
    
    # Resize to target size
    data = tf.image.resize(data, [image_size, image_size])
    
    if is_training:
        # Random horizontal flip (simulates temporal variation)
        data = tf.image.random_flip_left_right(data)
        
        # Add Gaussian noise (with probability)
        data = add_gaussian_noise(data, noise_level)
        
        # Random brightness (optional, simulates amplitude variation)
        data = tf.image.random_brightness(data, max_delta=0.2)
        
        # Clip to valid range after augmentation
        data = tf.clip_by_value(data, -1.0, 1.0)
    
    return data


# ===============================
# EXAMPLE USAGE
# ===============================
def create_augmentation_pipeline(spectrograms, labels, batch_size=32, 
                                image_size=224, is_training=True):
    """
    Create a tf.data pipeline with augmentation.
    
    Args:
        spectrograms: numpy array of spectrograms
        labels: numpy array of labels
        batch_size: batch size
        image_size: target image size
        is_training: whether this is for training
    """
    dataset = tf.data.Dataset.from_tensor_slices((spectrograms, labels))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=1000)
    
    def augment_fn(spec, label):
        spec = preprocess_cough_spectrogram(
            spec, 
            image_size=image_size,
            is_training=is_training
        )
        return spec, label
    
    dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset