# -*- coding: utf-8 -*-
"""
Live Detection Script (YAMNet Transfer Learning Version)
"""

import os
import sys
import datetime
import pickle
import numpy as np
import sounddevice as sd
import soundfile as sf
import librosa
import tensorflow as tf
import tensorflow_hub as hub
import serial
import time

# TensorFlow log seviyesini ayarlaa
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# =============================================================================
# AYARLAR
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "yamnet_transfer_model.h5")
ENCODER_PATH = os.path.join(SCRIPT_DIR, "yamnet_encoder.pkl")
RECORDINGS_DIR = os.path.join(SCRIPT_DIR, "live_recordings")

# YAMNet Native Handle
YAMNET_MODEL_HANDLE = 'https://tfhub.dev/google/yamnet/1'

# YAMNet Parametreleri
SAMPLE_RATE = 16000 # YAMNet 16k zorunlu
DURATION = 5        # 5 saniyelik dinleme
CONFIDENCE_THRESHOLD = 40.0 # Transfer learning daha katı olabilir, eşiği ayarladık
RMS_THRESHOLD = 0.005 

# Arduino Ayarları
ARDUINO_PORT = 'COM9'
ARDUINO_BAUD = 9600

# Etiket Çevirileri
LABEL_TR = {
    "hungry": "Açlık 🍼",
    "belly_pain": "Karın Ağrısı 😣",
    "burping": "Gaz/Geğirme 💨",
    "discomfort": "Rahatsızlık 😫",
    "tired": "Yorgunluk 😴"
}

if not os.path.exists(RECORDINGS_DIR):
    os.makedirs(RECORDINGS_DIR)

# =============================================================================
# YARDIMCI FONKSİYONLAR
# =============================================================================

def load_components():
    """Modelleri yükle"""
    print("Modeller yükleniyor (Biraz sürebilir)...")
    try:
        # 1. YAMNet Yükle
        print("  - YAMNet indiriliyor/yükleniyor...")
        yamnet = hub.load(YAMNET_MODEL_HANDLE)
        
        # 2. Bizim Sınıflandırıcıyı Yükle
        print("  - Sınıflandırıcı yükleniyor...")
        classifier = tf.keras.models.load_model(MODEL_PATH)
        
        # 3. Encoder Yükle
        with open(ENCODER_PATH, 'rb') as f:
            encoder = pickle.load(f)
            
        print("✅ Tüm modeller hazır.")
        return yamnet, classifier, encoder
    except Exception as e:
        print(f"❌ Model yükleme hatası: {e}")
        print("Lütfen önce 'train_transfer.py' çalıştırdığınızdan emin olun.")
        sys.exit(1)

def extract_embedding(yamnet, audio_data):
    """Sesten YAMNet özetini çıkar"""
    # Normalizasyon
    waveform = audio_data / np.max(np.abs(audio_data) + 1e-9)
    
    # YAMNet Çalıştır
    # Çıktılar: scores, embeddings, spectrogram
    _, embeddings, _ = yamnet(waveform)
    
    # Global Average Pooling (Tüm zamanların ortalaması)
    global_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
    
    # Model (1, 1024) bekliyor
    return global_embedding.reshape(1, -1)

def save_recording(audio, fs, filename_prefix="rec"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.wav"
    filepath = os.path.join(RECORDINGS_DIR, filename)
    sf.write(filepath, audio, fs)
    return filepath

def connect_arduino():
    """Arduino'ya bağlan"""
    try:
        arduino = serial.Serial(ARDUINO_PORT, ARDUINO_BAUD, timeout=1)
        time.sleep(2)  # Arduino reset bekle
        print(f"✅ Arduino bağlandı ({ARDUINO_PORT})")
        return arduino
    except Exception as e:
        print(f"⚠️ Arduino bağlantı hatası: {e}")
        print("   LCD olmadan devam ediliyor...")
        return None

def send_to_arduino(arduino, label, confidence):
    """Sonucu Arduino'ya gönder"""
    if arduino is None:
        return
    try:
        # LCD için Türkçe karakter düzeltme
        lcd_text = label.replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
        lcd_text = lcd_text.replace("İ", "I").replace("Ğ", "G").replace("Ü", "U").replace("Ş", "S").replace("Ö", "O").replace("Ç", "C")
        # Emoji kaldır
        for emoji in ['🍼', '😣', '💨', '😫', '😴']:
            lcd_text = lcd_text.replace(emoji, '')
        lcd_text = lcd_text.strip()
        
        # İki satır: Üst satır sebep, alt satır güven
        message = f"{lcd_text[:16]}%{confidence:.0f} Guven"
        arduino.write(f"{message}\n".encode('ascii', errors='ignore'))
        print(f"📟 LCD'ye gönderildi: {lcd_text}")
    except Exception as e:
        print(f"⚠️ Arduino gönderim hatası: {e}")

def select_microphone():
    print("\n🎧 MİKROFON SEÇİMİ")
    print("-" * 30)
    devices = sd.query_devices()
    input_devices = []
    
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            input_devices.append((i, device))
            print(f"[{i}] {device['name']}")
    
    if not input_devices:
        print("❌ Hiçbir mikrofon bulunamadı!")
        sys.exit(1)
        
    print("-" * 30)
    
    while True:
        try:
            selection = input("Lütfen mikrofon numarasını girin (Varsayılan için Enter): ")
            if selection.strip() == "":
                return None
            idx = int(selection)
            valid_indices = [d[0] for d in input_devices]
            if idx in valid_indices:
                return idx
            print("❌ Geçersiz numara.")
        except ValueError:
            print("❌ Sayı girin.")

def print_prediction_bar(all_probs, classes, predicted_idx):
    print("\nDETAYLI ANALİZ:")
    probs_with_labels = [(classes[i], all_probs[i]*100) for i in range(len(classes))]
    probs_with_labels.sort(key=lambda x: x[1], reverse=True)
    
    for label, prob in probs_with_labels:
        bar_len = int(prob / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        tr = LABEL_TR.get(label, label)
        prefix = "👉" if label == classes[predicted_idx] else "  "
        print(f"  {prefix} {tr:15} [{bar}] {prob:5.1f}%")

# =============================================================================
# MAIN
# =============================================================================
def main():
    yamnet, classifier, encoder = load_components()
    arduino = connect_arduino()
    
    device_index = select_microphone()
    block_size = int(SAMPLE_RATE * DURATION)
    classes = encoder.classes_
    
    print("\n" + "="*60)
    print(f"🎤 GELİŞMİŞ BEBEK AĞLAMASI ALGILAYICI (YAMNet)")
    print(f"⏱️  Kayıt Süresi: {DURATION} sn")
    print(f"�️  Güven Eşiği: %{CONFIDENCE_THRESHOLD}")
    print("="*60 + "\n")
    
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=block_size, device=device_index) as stream:
            while True:
                print(f"⏳ {DURATION}sn dinleniyor...", end="\r")
                
                audio_data, _ = stream.read(block_size)
                audio_np = audio_data.flatten()
                
                # RMS Kontrol
                rms = np.sqrt(np.mean(audio_np**2))
                
                if rms < RMS_THRESHOLD:
                    print(f"🔇 Çok Sesiz (RMS: {rms:.4f})                ", end="\r")
                    continue
                
                # Kaydet
                save_recording(audio_np, SAMPLE_RATE, "detected_yamnet")
                print(f"\n\n📢 SES ALGILANDI (RMS: {rms:.4f})")
                
                try:
                    # 1. YAMNet'ten geçir
                    embedding = extract_embedding(yamnet, audio_np)
                    
                    # 2. Sınıflandır
                    prediction = classifier.predict(embedding, verbose=0)[0]
                    
                    predicted_index = np.argmax(prediction)
                    confidence = prediction[predicted_index] * 100
                    
                    if confidence < CONFIDENCE_THRESHOLD:
                        print(f"⚠️  Düşük Güven (%{confidence:.1f}).")
                        print_prediction_bar(prediction, classes, predicted_index)
                    else:
                        predicted_label = encoder.inverse_transform([predicted_index])[0]
                        tr_label = LABEL_TR.get(predicted_label, predicted_label)
                        
                        print(f"🎯 TESPİT: {tr_label}")
                        print(f"✅ Güven:  %{confidence:.1f}")
                        print_prediction_bar(prediction, classes, predicted_index)
                        
                        # Arduino'ya gönder
                        send_to_arduino(arduino, tr_label, confidence)
                        
                    print("-" * 50)
                    
                except Exception as e:
                    print(f"❌ Analiz Hatası: {e}")
                    import traceback
                    traceback.print_exc()

    except KeyboardInterrupt:
        print("\n🛑 Çıkış.")

if __name__ == "__main__":
    main()
