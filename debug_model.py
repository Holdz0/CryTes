# -*- coding: utf-8 -*-
"""
Debug Script - Model ve Özellik Çıkarma Kontrolü
"""

import os
import pickle
import numpy as np
import librosa

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "baby_cry_model.h5")
ENCODER_PATH = os.path.join(SCRIPT_DIR, "label_encoder.pkl")
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset")

N_MFCC = 40
SAMPLE_RATE = 8000

print("="*60)
print("DEBUG: Model ve Özellik Çıkarma Kontrolü")
print("="*60)

# 1. Model yükle
print("\n[1] Model yükleniyor...")
model = tf.keras.models.load_model(MODEL_PATH)
print(f"    Model input shape: {model.input_shape}")
print(f"    Model output shape: {model.output_shape}")

# 2. Model ağırlıklarını kontrol et
print("\n[2] Model ağırlıkları kontrol ediliyor...")
for i, layer in enumerate(model.layers):
    weights = layer.get_weights()
    if weights:
        w = weights[0]
        print(f"    Layer {i} ({layer.name}): shape={w.shape}, mean={w.mean():.6f}, std={w.std():.6f}")

# 3. Farklı dosyalardan özellik çıkar ve karşılaştır
print("\n[3] Farklı dosyalardan MFCC özellikleri çıkarılıyor...")
test_files = [
    ("hungry", "hn_001.wav"),
    ("hungry", "hn_050.wav"),
    ("belly_pain", "bp_001.wav"),
    ("burping", "bu_001.wav"),
    ("tired", "ti_001.wav"),
]

features_list = []
for class_name, file_name in test_files:
    file_path = os.path.join(DATASET_PATH, class_name, file_name)
    if os.path.exists(file_path):
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC)
        features = np.mean(mfccs, axis=1)
        features_list.append(features)
        print(f"    {class_name}/{file_name}:")
        print(f"        Audio length: {len(audio)}, SR: {sr}")
        print(f"        MFCC shape: {mfccs.shape}")
        print(f"        Features (first 5): {features[:5]}")
        print(f"        Features mean: {features.mean():.4f}, std: {features.std():.4f}")
    else:
        print(f"    [HATA] Dosya bulunamadı: {file_path}")

# 4. Özelliklerin birbirinden farklı olup olmadığını kontrol et
print("\n[4] Özellik vektörleri farklı mı?")
if len(features_list) >= 2:
    for i in range(len(features_list)-1):
        diff = np.abs(features_list[i] - features_list[i+1]).mean()
        print(f"    Dosya {i} vs Dosya {i+1}: Ortalama fark = {diff:.6f}")

# 5. Tahminleri kontrol et
print("\n[5] Model tahminleri test ediliyor...")
for i, (features, (class_name, file_name)) in enumerate(zip(features_list, test_files)):
    features_reshaped = features.reshape(1, -1)
    predictions = model.predict(features_reshaped, verbose=0)[0]
    print(f"    {class_name}/{file_name}:")
    print(f"        Predictions: {predictions}")
    print(f"        Predictions unique: {len(np.unique(np.round(predictions, 4)))}")

# 6. Encoder kontrol
print("\n[6] Label Encoder kontrol...")
with open(ENCODER_PATH, 'rb') as f:
    encoder = pickle.load(f)
print(f"    Classes: {encoder.classes_}")

print("\n" + "="*60)
print("DEBUG TAMAMLANDI")
print("="*60)
