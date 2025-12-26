# -*- coding: utf-8 -*-
"""
Bebek Ağlaması Sınıflandırma - Test Scripti (v3 - CNN Uyumlu)

Kullanım: 
    python test_model.py                           # Dataset'ten örnekler test et
    python test_model.py dosya.wav                 # Tek dosya test et
    python test_model.py klasor/                   # Klasördeki wav'ları test et
"""

import os
import sys
import pickle
import numpy as np
import librosa

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

# =============================================================================
# KONFİGÜRASYON
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "yamnet_transfer_model.h5")
ENCODER_PATH = os.path.join(SCRIPT_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(SCRIPT_DIR, "scaler.pkl")
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset")

N_MFCC = 40
SAMPLE_RATE = 8000
MAX_PAD_LEN = 174  # CNN modeli için sabit boyut

# Türkçe etiket çevirileri
LABEL_TR = {
    "hungry": "Açlık 🍼",
    "belly_pain": "Karın Ağrısı 😣",
    "burping": "Gaz/Geğirme 💨",
    "discomfort": "Rahatsızlık 😫",
    "tired": "Yorgunluk 😴"
}

# =============================================================================
# ÖZELLİK ÇIKARMA
# =============================================================================
def extract_mfcc_1d(audio, sr, n_mfcc=N_MFCC):
    """1D özellik çıkarma (Dense model için)."""
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)


def extract_mfcc_2d(audio, sr, n_mfcc=N_MFCC, max_pad_len=MAX_PAD_LEN):
    """2D özellik çıkarma (CNN model için)."""
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Padding veya kırpma
    if mfccs.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfccs = mfccs[:, :max_pad_len]
    
    return mfccs.T  # (time, features)


def load_components():
    """Model, encoder ve scaler'ı yükle."""
    if not os.path.exists(MODEL_PATH):
        print(f"[HATA] Model bulunamadı: {MODEL_PATH}")
        sys.exit(1)
    
    if not os.path.exists(ENCODER_PATH):
        print(f"[HATA] Encoder bulunamadı: {ENCODER_PATH}")
        sys.exit(1)
    
    if not os.path.exists(SCALER_PATH):
        print(f"[HATA] Scaler bulunamadı: {SCALER_PATH}")
        sys.exit(1)
    
    model = tf.keras.models.load_model(MODEL_PATH)
    
    with open(ENCODER_PATH, 'rb') as f:
        encoder = pickle.load(f)
    
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
    
    # Model tipini belirle
    input_shape = model.input_shape
    is_cnn = len(input_shape) == 3  # (None, time, features)
    
    return model, encoder, scaler, is_cnn


def predict_single(model, encoder, scaler, file_path, is_cnn):
    """Tek dosya tahmini."""
    try:
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
        if len(audio) == 0:
            return None, None, None
    except Exception as e:
        print(f"[HATA] {file_path}: {e}")
        return None, None, None
    
    # Özellik çıkar
    if is_cnn:
        features = extract_mfcc_2d(audio, sr)
        features_flat = features.reshape(-1, features.shape[-1])
        features_scaled = scaler.transform(features_flat)
        features_final = features_scaled.reshape(1, features.shape[0], features.shape[1])
    else:
        features = extract_mfcc_1d(audio, sr)
        features_scaled = scaler.transform(features.reshape(1, -1))
        features_final = features_scaled
    
    # Tahmin
    predictions = model.predict(features_final, verbose=0)[0]
    predicted_idx = np.argmax(predictions)
    predicted_class = encoder.inverse_transform([predicted_idx])[0]
    confidence = predictions[predicted_idx] * 100
    
    return predicted_class, confidence, predictions


def print_prediction(file_path, predicted_class, confidence, all_probs, encoder):
    """Tahmin sonucunu yazdır."""
    tr_label = LABEL_TR.get(predicted_class, predicted_class)
    
    print(f"\n{'='*60}")
    print(f"📁 Dosya: {os.path.basename(file_path)}")
    print(f"{'='*60}")
    print(f"🎯 TAHMİN: {tr_label}")
    print(f"📊 Güven:  {confidence:.2f}%")
    print(f"\n📈 Tüm Olasılıklar:")
    
    classes = encoder.classes_
    probs_with_labels = [(classes[i], all_probs[i]*100) for i in range(len(classes))]
    probs_with_labels.sort(key=lambda x: x[1], reverse=True)
    
    for label, prob in probs_with_labels:
        bar = "█" * int(prob / 5) + "░" * (20 - int(prob / 5))
        tr = LABEL_TR.get(label, label)
        marker = " ← " if label == predicted_class else "   "
        print(f"  {tr:20} [{bar}] {prob:5.1f}%{marker}")


def test_dataset_samples(model, encoder, scaler, is_cnn, num_per_class=5):
    """Dataset'ten örnekler test et."""
    print("\n" + "="*60)
    print("   DATASET TEST")
    print("="*60)
    
    total_correct = 0
    total_tested = 0
    
    import random
    
    for class_name in sorted(os.listdir(DATASET_PATH)):
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.isdir(class_path):
            continue
        
        wav_files = [f for f in os.listdir(class_path) if f.endswith('.wav')]
        if not wav_files:
            continue
        
        samples = random.sample(wav_files, min(num_per_class, len(wav_files)))
        
        print(f"\n📁 {LABEL_TR.get(class_name, class_name)}")
        print("-" * 50)
        
        correct = 0
        for wav_file in samples:
            file_path = os.path.join(class_path, wav_file)
            result = predict_single(model, encoder, scaler, file_path, is_cnn)
            
            if result[0] is None:
                continue
            
            predicted, confidence, _ = result
            is_correct = predicted == class_name
            if is_correct:
                correct += 1
                total_correct += 1
            total_tested += 1
            
            status = "✅" if is_correct else "❌"
            pred_tr = LABEL_TR.get(predicted, predicted)
            print(f"  {status} {wav_file[:25]:25} → {pred_tr:15} ({confidence:.1f}%)")
        
        print(f"  📊 Doğruluk: {correct}/{len(samples)} ({100*correct/len(samples):.0f}%)")
    
    print(f"\n{'='*60}")
    print(f"🎯 GENEL DOĞRULUK: {total_correct}/{total_tested} ({100*total_correct/total_tested:.1f}%)")
    print(f"{'='*60}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*60)
    print("   BEBEK AĞLAMASI SINIFLANDIRICI - TEST (v3)")
    print("="*60)
    
    print("\n📥 Bileşenler yükleniyor...")
    model, encoder, scaler, is_cnn = load_components()
    print(f"✅ Model tipi: {'CNN' if is_cnn else 'Dense'}")
    print(f"✅ Sınıflar: {list(encoder.classes_)}")
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
        
        if os.path.isfile(input_path) and input_path.endswith('.wav'):
            result = predict_single(model, encoder, scaler, input_path, is_cnn)
            if result[0]:
                print_prediction(input_path, result[0], result[1], result[2], encoder)
        
        elif os.path.isdir(input_path):
            wav_files = [f for f in os.listdir(input_path) if f.endswith('.wav')]
            print(f"\n📂 {len(wav_files)} dosya test ediliyor...")
            
            for wav_file in wav_files[:10]:  # Max 10 dosya
                file_path = os.path.join(input_path, wav_file)
                result = predict_single(model, encoder, scaler, file_path, is_cnn)
                if result[0]:
                    print_prediction(file_path, result[0], result[1], result[2], encoder)
        else:
            print(f"[HATA] Geçersiz yol: {input_path}")
    else:
        test_dataset_samples(model, encoder, scaler, is_cnn, num_per_class=20)
    
    print("\n" + "="*60)
    print("   TEST TAMAMLANDI")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
