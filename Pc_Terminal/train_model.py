# -*- coding: utf-8 -*-
"""
Bebek Ağlaması Neden Sınıflandırma - Model Eğitim Scripti (v2 - Fixed)
(Baby Cry Reason Classifier - Training Script)

Kullanım: python train_model.py

Gerekli Kütüphaneler (kurulum):
# pip install tensorflow librosa numpy scikit-learn matplotlib

Yazar: AI Mühendis
Tarih: 2025-12-26
Düzeltme: StandardScaler ile normalizasyon eklendi
"""

# =============================================================================
# 1. KÜTÜPHANE İMPORTLARI
# =============================================================================
import os
import warnings
import pickle
import numpy as np
import librosa
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# TensorFlow uyarılarını bastır
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# =============================================================================
# 2. KONFİGÜRASYON
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset")

# Model çıktı dosyaları
MODEL_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "baby_cry_model.h5")
ENCODER_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "label_encoder.pkl")
SCALER_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "scaler.pkl")  # YENİ: Scaler kaydet
HISTORY_PLOT_PATH = os.path.join(SCRIPT_DIR, "training_history.png")

# MFCC parametreleri
N_MFCC = 40
SAMPLE_RATE = 8000

# Eğitim parametreleri
TEST_SIZE = 0.2
RANDOM_STATE = 42
EPOCHS = 150
BATCH_SIZE = 16  # Daha küçük batch size

# =============================================================================
# 3. VERİ YÜKLEME FONKSİYONLARI
# =============================================================================
def extract_mfcc_features(file_path, n_mfcc=N_MFCC, sr=SAMPLE_RATE):
    """
    Bir ses dosyasından MFCC özelliklerini çıkarır.
    """
    try:
        audio, sample_rate = librosa.load(file_path, sr=sr, mono=True)
        
        if len(audio) == 0:
            print(f"[UYARI] Boş ses dosyası atlandı: {file_path}")
            return None
        
        # MFCC özelliklerini çıkar
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        
        # Zaman ekseni boyunca ortalama al
        mfccs_mean = np.mean(mfccs, axis=1)
        
        return mfccs_mean
        
    except Exception as e:
        print(f"[HATA] Dosya yüklenemedi: {file_path} - {str(e)}")
        return None


def load_dataset(dataset_path):
    """Dataset klasöründen tüm ses dosyalarını yükler."""
    features = []
    labels = []
    
    total_files = 0
    class_dirs = []
    
    for class_name in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_path):
            wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
            if wav_files:
                class_dirs.append((class_name, class_path, wav_files))
                total_files += len(wav_files)
    
    print(f"\n{'='*60}")
    print(f"DATASET BİLGİLERİ")
    print(f"{'='*60}")
    print(f"Toplam sınıf sayısı: {len(class_dirs)}")
    print(f"Toplam dosya sayısı: {total_files}")
    print(f"Sınıflar: {[c[0] for c in class_dirs]}")
    print(f"{'='*60}\n")
    
    processed = 0
    skipped = 0
    
    for class_name, class_path, wav_files in class_dirs:
        print(f"\n[İŞLENİYOR] Sınıf: {class_name} ({len(wav_files)} dosya)")
        
        class_processed = 0
        for wav_file in wav_files:
            file_path = os.path.join(class_path, wav_file)
            mfcc_features = extract_mfcc_features(file_path)
            
            if mfcc_features is not None:
                features.append(mfcc_features)
                labels.append(class_name)
                class_processed += 1
            else:
                skipped += 1
            
            processed += 1
            
            if processed % 100 == 0:
                print(f"    İlerleme: {processed}/{total_files} ({100*processed/total_files:.1f}%)")
        
        print(f"    ✓ {class_name}: {class_processed} dosya başarıyla işlendi")
    
    print(f"\n{'='*60}")
    print(f"YÜKLEME TAMAMLANDI")
    print(f"{'='*60}")
    print(f"Başarılı: {len(features)} dosya")
    print(f"Atlanan:  {skipped} dosya")
    print(f"{'='*60}\n")
    
    return np.array(features), np.array(labels)


# =============================================================================
# 4. MODEL OLUŞTURMA FONKSİYONU
# =============================================================================
def create_model(input_shape, num_classes):
    """Geliştirilmiş model mimarisi."""
    model = Sequential([
        Input(shape=input_shape),
        
        # İlk gizli katman
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # İkinci gizli katman
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        
        # Üçüncü gizli katman (ekstra)
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        # Çıkış katmanı
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# =============================================================================
# 5. EĞİTİM GEÇMİŞİ GRAFİĞİ
# =============================================================================
def plot_training_history(history, output_path):
    """Eğitim grafiklerini çizer."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['accuracy'], label='Eğitim', linewidth=2, color='#2196F3')
    axes[0].plot(history.history['val_accuracy'], label='Doğrulama', linewidth=2, color='#4CAF50')
    axes[0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['loss'], label='Eğitim', linewidth=2, color='#F44336')
    axes[1].plot(history.history['val_loss'], label='Doğrulama', linewidth=2, color='#FF9800')
    axes[1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n[✓] Eğitim grafikleri kaydedildi: {output_path}")


# =============================================================================
# 6. ANA EĞİTİM FONKSİYONU
# =============================================================================
def main():
    print("\n" + "="*60)
    print("   BEBEK AĞLAMASI SINIFLANDIRMA - MODEL EĞİTİMİ (v2)")
    print("="*60)
    
    # Dataset kontrolü
    if not os.path.exists(DATASET_PATH):
        print(f"\n[HATA] Dataset klasörü bulunamadı: {DATASET_PATH}")
        return
    
    # Veriyi yükle
    print("\n[1/6] Ses dosyaları yükleniyor...")
    X, y = load_dataset(DATASET_PATH)
    
    if len(X) == 0:
        print("[HATA] Hiçbir ses dosyası yüklenemedi!")
        return
    
    print(f"Özellik boyutu: {X.shape}")
    
    # ===== KRİTİK: NORMALIZASYON =====
    print("\n[2/6] Özellikler normalize ediliyor (StandardScaler)...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"    Normalizasyon öncesi - Mean: {X.mean():.4f}, Std: {X.std():.4f}")
    print(f"    Normalizasyon sonrası - Mean: {X_scaled.mean():.4f}, Std: {X_scaled.std():.4f}")
    
    # Scaler'ı kaydet
    with open(SCALER_OUTPUT_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"    ✓ Scaler kaydedildi: {SCALER_OUTPUT_PATH}")
    
    # Etiketleri encode et
    print("\n[3/6] Etiketler encode ediliyor...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    classes = label_encoder.classes_
    print(f"Sınıflar: {list(classes)}")
    
    y_categorical = to_categorical(y_encoded, num_classes=len(classes))
    
    # LabelEncoder'ı kaydet
    with open(ENCODER_OUTPUT_PATH, 'wb') as f:
        pickle.dump(label_encoder, f)
    print(f"    ✓ LabelEncoder kaydedildi: {ENCODER_OUTPUT_PATH}")
    
    # Train/Test split
    print("\n[4/6] Veri train/test olarak ayrılıyor...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_categorical, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y_encoded
    )
    
    print(f"Eğitim seti: {X_train.shape[0]} örnek")
    print(f"Test seti:   {X_test.shape[0]} örnek")
    
    # Model oluştur
    print("\n[5/6] Model oluşturuluyor...")
    model = create_model(
        input_shape=(N_MFCC,),
        num_classes=len(classes)
    )
    
    print("\nModel Mimarisi:")
    model.summary()
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=0.00001,
        verbose=1
    )
    
    # Eğitim
    print("\n[6/6] Model eğitimi başlıyor...")
    print("-" * 60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    # Değerlendirme
    print("\n" + "="*60)
    print("MODEL DEĞERLENDİRMESİ")
    print("="*60)
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss:     {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Model kaydet
    model.save(MODEL_OUTPUT_PATH)
    print(f"\n[✓] Model kaydedildi: {MODEL_OUTPUT_PATH}")
    
    # Grafik kaydet
    plot_training_history(history, HISTORY_PLOT_PATH)
    
    # Özet
    print("\n" + "="*60)
    print("EĞİTİM TAMAMLANDI - ÖZET")
    print("="*60)
    print(f"Model:        {MODEL_OUTPUT_PATH}")
    print(f"Encoder:      {ENCODER_OUTPUT_PATH}")
    print(f"Scaler:       {SCALER_OUTPUT_PATH}")
    print(f"Grafikler:    {HISTORY_PLOT_PATH}")
    print(f"Epoch sayısı: {len(history.history['loss'])}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"Sınıflar:     {list(classes)}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
