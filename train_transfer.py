# -*- coding: utf-8 -*-
"""
Bebek Ağlaması Sınıflandırma - Transfer Learning (YAMNet)
"""

import os
import numpy as np
import pickle
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt

# =============================================================================
# AYARLAR
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset")
MODEL_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "yamnet_transfer_model.h5")
ENCODER_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "yamnet_encoder.pkl")
HISTORY_PLOT_PATH = os.path.join(SCRIPT_DIR, "yamnet_history.png")

# YAMNet Parametreleri
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'
SAMPLE_RATE = 16000  # YAMNet kesinlikle 16k istiyor

# Eğitim Parametreleri
EPOCHS = 100
BATCH_SIZE = 32

def load_yamnet_model():
    """YAMNet modelini TF Hub'dan yükle"""
    print("YAMNet modeli yükleniyor...")
    model = hub.load(YAMNET_MODEL_HANDLE)
    return model

def extract_yamnet_embeddings(yamnet_model, audio_path):
    """
    Ses dosyasından YAMNet embedding'leri çıkarır.
    YAMNet her 0.48s için bir embedding (1024 boyutlu) döndürür.
    Biz bunların ortalamasını alarak dosya başına tek bir vektör (1024,) elde edeceğiz.
    """
    try:
        # 16k Hz olarak yükle (YAMNet zorunluluğu)
        wav_data, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        
        # Sinyal çok kısaysa padding yap (en az 0.48s olmalı)
        if len(wav_data) < int(0.48 * SAMPLE_RATE):
            wav_data = np.pad(wav_data, (0, int(0.48 * SAMPLE_RATE) - len(wav_data)))
            
        # Normalizasyon (-1 ile 1 arasına)
        wav_data = wav_data / np.max(np.abs(wav_data))
        
        # YAMNet'e gönder
        # YAMNet çıktıları: scores, embeddings, spectrogram
        # Biz sadece embeddings (None, 1024) ile ilgileniyoruz
        _, embeddings, _ = yamnet_model(wav_data)
        
        # Tüm frame'lerin ortalamasını al (Global Average Pooling mantığı)
        # Böylece dosya uzunluğu ne olursa olsun çıktı (1024,) olur.
        global_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        
        return global_embedding
        
    except Exception as e:
        print(f"HATA: {audio_path} - {e}")
        return None

def prepare_dataset(yamnet_model):
    """Dataseti tara ve embeddingleri çıkar"""
    X = []
    y = []
    
    print("\nVeri seti hazırlanıyor ve YAMNet'ten geçiriliyor...")
    
    classes = [d for d in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, d))]
    
    total_files = sum([len(os.listdir(os.path.join(DATASET_PATH, c))) for c in classes])
    processed = 0
    
    for class_name in classes:
        class_dir = os.path.join(DATASET_PATH, class_name)
        files = [f for f in os.listdir(class_dir) if f.endswith('.wav')]
        
        print(f"  📂 Sınıf: {class_name} ({len(files)} dosya)")
        
        for f in files:
            file_path = os.path.join(class_dir, f)
            embedding = extract_yamnet_embeddings(yamnet_model, file_path)
            
            if embedding is not None:
                X.append(embedding)
                y.append(class_name)
            
            processed += 1
            print(f"    İlerleme: {processed}/{total_files}", end="\r")
            
    print(f"\n✅ Veri hazırlığı tamam. Toplam örnek: {len(X)}")
    return np.array(X), np.array(y)

def create_transfer_model(num_classes):
    """YAMNet embeddings üzerine eklenecek classifier"""
    model = models.Sequential([
        layers.Input(shape=(1024,)), # YAMNet çıktı boyutu
        
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.4),
        
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
        loss='sparse_categorical_crossentropy', # Label encoder integer verdiği için sparse
        metrics=['accuracy']
    )
    return model

def main():
    # 1. YAMNet Yükle
    yamnet = load_yamnet_model()
    
    # 2. Veriyi Hazırla (Embeddings)
    X, y = prepare_dataset(yamnet)
    
    # 3. Label Encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Encoder Kaydet
    with open(ENCODER_OUTPUT_PATH, 'wb') as f:
        pickle.dump(le, f)
    
    # Class Weights (Dengesiz veri için)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\nClass Weights: {class_weight_dict}")
    
    # 4. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # 5. Model Oluştur ve Eğit
    model = create_transfer_model(len(le.classes_))
    model.summary()
    
    early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=[early_stop],
        verbose=1
    )
    
    # 6. Kaydet
    model.save(MODEL_OUTPUT_PATH)
    print(f"\n✅ Model kaydedildi: {MODEL_OUTPUT_PATH}")
    
    # Değerlendirme
    loss, acc = model.evaluate(X_test, y_test)
    print(f"\n🎯 TEST ACCURACY: {acc*100:.2f}%")
    
    # Grafik
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title(f'Transfer Learning Accuracy (Test: {acc*100:.1f}%)')
    plt.savefig(HISTORY_PLOT_PATH)
    print("Grafik kaydedildi.")

if __name__ == "__main__":
    main()
