# -*- coding: utf-8 -*-
"""
Bebek Ağlaması Sınıflandırma - Gelişmiş Model (v3 - High Accuracy)
(Baby Cry Reason Classifier - Advanced Training Script)

Yenilikler:
- Data Augmentation (gürültü, zaman kaydırma, pitch değişimi)
- Gelişmiş özellikler (MFCC + Delta + Delta-Delta + Spectral)
- 1D CNN + Dense hibrit mimarisi
- Class weights (dengesiz veri için)

Kullanım: python train_model_v3.py

# pip install tensorflow librosa numpy scikit-learn matplotlib audiomentations
"""

import os
import warnings
import pickle
import numpy as np
import librosa
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, Input, BatchNormalization,
                                      Conv1D, MaxPooling1D, Flatten, GlobalAveragePooling1D)
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

# =============================================================================
# KONFİGÜRASYON
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset")

# Çıktı dosyaları
MODEL_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "baby_cry_model.h5")
ENCODER_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "label_encoder.pkl")
SCALER_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "scaler.pkl")
HISTORY_PLOT_PATH = os.path.join(SCRIPT_DIR, "training_history.png")

# Ses parametreleri
N_MFCC = 40
SAMPLE_RATE = 8000
MAX_PAD_LEN = 174  # Sabit zaman boyutu için

# Eğitim parametreleri
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 200
BATCH_SIZE = 32

# Data Augmentation ayarları
AUGMENTATION_FACTOR = 3  # Her örneği 3 kat artır

# =============================================================================
# DATA AUGMENTATION FONKSİYONLARI
# =============================================================================
def add_noise(audio, noise_factor=0.005):
    """Ses dosyasına gürültü ekle."""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

def time_shift(audio, shift_max=0.2):
    """Ses dosyasını zaman ekseninde kaydır."""
    shift = int(len(audio) * np.random.uniform(-shift_max, shift_max))
    return np.roll(audio, shift)

def pitch_shift(audio, sr, n_steps=None):
    """Pitch değiştir."""
    if n_steps is None:
        n_steps = np.random.randint(-3, 4)
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)

def time_stretch(audio, rate=None):
    """Hızı değiştir."""
    if rate is None:
        rate = np.random.uniform(0.8, 1.2)
    return librosa.effects.time_stretch(audio, rate=rate)

def augment_audio(audio, sr):
    """Rastgele augmentation uygula."""
    augmented = audio.copy()
    
    # Rastgele augmentation seç
    aug_type = np.random.choice(['noise', 'shift', 'pitch', 'stretch', 'combined'])
    
    if aug_type == 'noise':
        augmented = add_noise(augmented, noise_factor=np.random.uniform(0.002, 0.01))
    elif aug_type == 'shift':
        augmented = time_shift(augmented, shift_max=0.15)
    elif aug_type == 'pitch':
        augmented = pitch_shift(augmented, sr, n_steps=np.random.randint(-2, 3))
    elif aug_type == 'stretch':
        augmented = time_stretch(augmented, rate=np.random.uniform(0.9, 1.1))
    elif aug_type == 'combined':
        augmented = add_noise(augmented, noise_factor=0.003)
        augmented = time_shift(augmented, shift_max=0.1)
    
    return augmented


# =============================================================================
# GELİŞMİŞ ÖZELLİK ÇIKARMA
# =============================================================================
def extract_advanced_features(audio, sr, n_mfcc=N_MFCC):
    """
    Gelişmiş özellik çıkarma:
    - MFCC (40)
    - Delta MFCC (40)
    - Delta-Delta MFCC (40)
    - Spectral Centroid, Rolloff, ZCR (3)
    Toplam: 123 özellik
    """
    features = []
    
    # MFCC
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    features.extend(mfccs_mean)
    
    # Delta MFCC (1. türev)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta_mean = np.mean(delta_mfccs, axis=1)
    features.extend(delta_mean)
    
    # Delta-Delta MFCC (2. türev)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    delta2_mean = np.mean(delta2_mfccs, axis=1)
    features.extend(delta2_mean)
    
    # Spectral Centroid
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    features.append(np.mean(spec_centroid))
    
    # Spectral Rolloff
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
    features.append(np.mean(spec_rolloff))
    
    # Zero Crossing Rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    features.append(np.mean(zcr))
    
    # RMS Energy
    rms = librosa.feature.rms(y=audio)
    features.append(np.mean(rms))
    
    # Spectral Bandwidth
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
    features.append(np.mean(spec_bw))
    
    return np.array(features)


def extract_mfcc_2d(audio, sr, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    """
    CNN için 2D MFCC çıkar (zaman boyutunu koru).
    """
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Padding veya kırpma
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    
    return mfccs.T  # (time, features) şeklinde döndür


# =============================================================================
# VERİ YÜKLEME
# =============================================================================
def load_dataset_with_augmentation(dataset_path, use_2d=True, augment=True):
    """
    Dataset'i yükle ve data augmentation uygula.
    """
    features = []
    labels = []
    
    # Sınıfları bul
    class_dirs = []
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
            if wav_files:
                class_dirs.append((class_name, class_path, wav_files))
    
    total_original = sum(len(c[2]) for c in class_dirs)
    print(f"\n{'='*60}")
    print(f"DATASET BİLGİLERİ")
    print(f"{'='*60}")
    print(f"Sınıf sayısı: {len(class_dirs)}")
    print(f"Orijinal dosya: {total_original}")
    if augment:
        print(f"Augmentation sonrası (tahmini): ~{total_original * (AUGMENTATION_FACTOR + 1)}")
    print(f"{'='*60}\n")
    
    processed = 0
    
    for class_name, class_path, wav_files in class_dirs:
        print(f"[İŞLENİYOR] {class_name} ({len(wav_files)} dosya)")
        
        for wav_file in wav_files:
            file_path = os.path.join(class_path, wav_file)
            
            try:
                audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
                
                if len(audio) == 0:
                    continue
                
                # Orijinal örnek
                if use_2d:
                    feat = extract_mfcc_2d(audio, sr)
                else:
                    feat = extract_advanced_features(audio, sr)
                
                features.append(feat)
                labels.append(class_name)
                
                # Augmented örnekler
                if augment:
                    for _ in range(AUGMENTATION_FACTOR):
                        aug_audio = augment_audio(audio, sr)
                        
                        if use_2d:
                            feat_aug = extract_mfcc_2d(aug_audio, sr)
                        else:
                            feat_aug = extract_advanced_features(aug_audio, sr)
                        
                        features.append(feat_aug)
                        labels.append(class_name)
                
            except Exception as e:
                print(f"    [HATA] {wav_file}: {e}")
            
            processed += 1
            if processed % 100 == 0:
                print(f"    İlerleme: {processed}/{total_original}")
        
        print(f"    ✓ Tamamlandı")
    
    print(f"\n✅ Toplam örnek: {len(features)}")
    
    return np.array(features), np.array(labels)


# =============================================================================
# MODEL MİMARİLERİ
# =============================================================================
def create_cnn_model(input_shape, num_classes):
    """
    1D CNN modeli (zaman serisi için optimized).
    """
    model = Sequential([
        Input(shape=input_shape),
        
        # Conv Block 1
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Conv Block 2
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        
        # Conv Block 3
        Conv1D(256, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        
        # Global Pooling
        GlobalAveragePooling1D(),
        
        # Dense Layers
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def create_dense_model(input_shape, num_classes):
    """
    Geliştirilmiş Dense model (1D özellikler için).
    """
    model = Sequential([
        Input(shape=input_shape),
        
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.4),
        
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# =============================================================================
# EĞİTİM GRAFİĞİ
# =============================================================================
def plot_training_history(history, output_path):
    """Eğitim grafiklerini çizer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], label='Eğitim', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Doğrulama', linewidth=2)
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Eğitim', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Doğrulama', linewidth=2)
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[✓] Grafik kaydedildi: {output_path}")


# =============================================================================
# ANA FONKSİYON
# =============================================================================
def main():
    print("\n" + "="*60)
    print("   BEBEK AĞLAMASI SINIFLANDIRMA - GELİŞMİŞ MODEL (v3)")
    print("   Data Augmentation + CNN + Advanced Features")
    print("="*60)
    
    # Model tipi seçimi
    USE_CNN = True  # True: CNN, False: Dense
    
    if not os.path.exists(DATASET_PATH):
        print(f"[HATA] Dataset bulunamadı: {DATASET_PATH}")
        return
    
    # Veri yükleme
    print("\n[1/6] Veri yükleniyor ve augmentation uygulanıyor...")
    X, y = load_dataset_with_augmentation(DATASET_PATH, use_2d=USE_CNN, augment=True)
    
    if len(X) == 0:
        print("[HATA] Veri yüklenemedi!")
        return
    
    print(f"\nVeri boyutu: {X.shape}")
    
    # Normalizasyon
    print("\n[2/6] Normalizasyon uygulanıyor...")
    original_shape = X.shape
    
    if USE_CNN:
        # CNN için 2D normalizasyon
        X_flat = X.reshape(-1, X.shape[-1])
        scaler = StandardScaler()
        X_flat_scaled = scaler.fit_transform(X_flat)
        X_scaled = X_flat_scaled.reshape(original_shape)
    else:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    
    # Scaler kaydet
    with open(SCALER_OUTPUT_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"    ✓ Scaler kaydedildi")
    
    # Etiket encoding
    print("\n[3/6] Etiketler encode ediliyor...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    classes = label_encoder.classes_
    num_classes = len(classes)
    
    print(f"    Sınıflar: {list(classes)}")
    print(f"    Sınıf dağılımı: {Counter(y)}")
    
    y_categorical = to_categorical(y_encoded, num_classes=num_classes)
    
    # Class weights hesapla (dengesiz veri için)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"    Class weights: {class_weight_dict}")
    
    # Encoder kaydet
    with open(ENCODER_OUTPUT_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"    ✓ Encoder kaydedildi")
    
    # Train/Test split
    print("\n[4/6] Train/Test ayrımı yapılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_categorical,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )
    print(f"    Eğitim: {X_train.shape[0]} örnek")
    print(f"    Test: {X_test.shape[0]} örnek")
    
    # Model oluştur
    print("\n[5/6] Model oluşturuluyor...")
    if USE_CNN:
        input_shape = (X_train.shape[1], X_train.shape[2])  # (time, features)
        model = create_cnn_model(input_shape, num_classes)
        print("    Model tipi: 1D CNN")
    else:
        input_shape = (X_train.shape[1],)
        model = create_dense_model(input_shape, num_classes)
        print("    Model tipi: Dense")
    
    model.summary()
    
    # Callbacks
    callbacks = [
        EarlyStopping(monitor='val_accuracy', patience=25, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1),
        ModelCheckpoint(MODEL_OUTPUT_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
    ]
    
    # Eğitim
    print("\n[6/6] Eğitim başlıyor...")
    print("-" * 60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    # Değerlendirme
    print("\n" + "="*60)
    print("MODEL DEĞERLENDİRMESİ")
    print("="*60)
    
    # En iyi modeli yükle
    model = tf.keras.models.load_model(MODEL_OUTPUT_PATH)
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Grafik
    plot_training_history(history, HISTORY_PLOT_PATH)
    
    # Özet
    print("\n" + "="*60)
    print("EĞİTİM TAMAMLANDI")
    print("="*60)
    print(f"Model:         {MODEL_OUTPUT_PATH}")
    print(f"Encoder:       {ENCODER_OUTPUT_PATH}")
    print(f"Scaler:        {SCALER_OUTPUT_PATH}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Model tipi:    {'CNN' if USE_CNN else 'Dense'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
