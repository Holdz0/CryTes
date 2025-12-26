# -*- coding: utf-8 -*-
"""
Bebek Ağlaması Test Scripti - YAMNet Transfer Learning
"""

import os
import sys
import pickle
import numpy as np
import librosa
import tensorflow as tf
import tensorflow_hub as hub

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================================
# AYARLAR
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "yamnet_transfer_model.h5")
ENCODER_PATH = os.path.join(SCRIPT_DIR, "yamnet_encoder.pkl")
DATASET_PATH = os.path.join(SCRIPT_DIR, "dataset")
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'

# Türkçe Etiketler
LABEL_TR = {
    "hungry": "Açlık 🍼",
    "belly_pain": "Karın Ağrısı 😣",
    "burping": "Gaz/Geğirme 💨",
    "discomfort": "Rahatsızlık 😫",
    "tired": "Yorgunluk 😴"
}

def load_components():
    print("Modeller Yükleniyor...")
    try:
        yamnet = hub.load(YAMNET_MODEL_HANDLE)
        classifier = tf.keras.models.load_model(MODEL_PATH)
        with open(ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
        return yamnet, classifier, encoder
    except Exception as e:
        print(f"❌ Hata: {e}")
        sys.exit(1)

def predict_file(yamnet, classifier, encoder, file_path):
    try:
        # YAMNet için 16k yükle
        audio, sr = librosa.load(file_path, sr=16000, mono=True)
        
        # Embedding çıkar
        waveform = audio / np.max(np.abs(audio) + 1e-9)
        _, embeddings, _ = yamnet(waveform)
        
        # Ortalama al (Global Pooling)
        global_embedding = tf.reduce_mean(embeddings, axis=0).numpy().reshape(1, -1)
        
        # Tahmin et
        predictions = classifier.predict(global_embedding, verbose=0)[0]
        predicted_idx = np.argmax(predictions)
        confidence = predictions[predicted_idx] * 100
        predicted_label = encoder.inverse_transform([predicted_idx])[0]
        
        return predicted_label, confidence, predictions
        
    except Exception as e:
        print(f"❌ Hata ({os.path.basename(file_path)}): {e}")
        return None, None, None

def print_result(file_path, label, confidence, all_probs, encoder):
    print(f"\n📂 Dosya: {os.path.basename(file_path)}")
    tr_label = LABEL_TR.get(label, label)
    print(f"🎯 TAHMİN: {tr_label}")
    print(f"📊 Güven:  {confidence:.2f}%")
    
    # Bar gösterimi
    print("-" * 40)
    classes = encoder.classes_
    probs_with_labels = [(classes[i], all_probs[i]*100) for i in range(len(classes))]
    probs_with_labels.sort(key=lambda x: x[1], reverse=True)
    
    for l, p in probs_with_labels:
        bar = "█" * int(p / 5) + "░" * (20 - int(p / 5))
        tr = LABEL_TR.get(l, l)
        mark = "👈" if l == label else "  "
        print(f"  {tr:15} [{bar}] {p:5.1f}% {mark}")
    print("-" * 40)

def test_dataset_sample(yamnet, classifier, encoder):
    print("\nDataset Testi Başlıyor (Her sınıftan 5 tane)...")
    import random
    
    total_correct = 0
    total_tested = 0
    
    for cls in encoder.classes_:
        cls_path = os.path.join(DATASET_PATH, cls)
        if not os.path.exists(cls_path): continue
        
        files = [f for f in os.listdir(cls_path) if f.endswith('.wav')]
        if not files: continue
        
        sample_files = random.sample(files, min(5, len(files)))
        
        print(f"\n📁 Sınıf: {LABEL_TR.get(cls, cls)}")
        
        correct = 0
        for f in sample_files:
            path = os.path.join(cls_path, f)
            pred, conf, _ = predict_file(yamnet, classifier, encoder, path)
            
            is_correct = (pred == cls)
            if is_correct: 
                correct += 1
                total_correct += 1
            total_tested += 1
            
            icon = "✅" if is_correct else "❌"
            print(f"  {icon} {f[:15]:15} -> {LABEL_TR.get(pred, pred)} (%{conf:.0f})")
            
    print(f"\n🎯 GENEL BAŞARI: {total_correct}/{total_tested} (%{total_correct/total_tested*100:.1f})")

def main():
    yamnet, classifier, encoder = load_components()
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            l, c, p = predict_file(yamnet, classifier, encoder, path)
            print_result(path, l, c, p, encoder)
        elif os.path.isdir(path):
            for f in os.listdir(path):
                if f.endswith('.wav'):
                    full_path = os.path.join(path, f)
                    l, c, p = predict_file(yamnet, classifier, encoder, full_path)
                    print_result(full_path, l, c, p, encoder)
    else:
        test_dataset_sample(yamnet, classifier, encoder)

if __name__ == "__main__":
    main()
