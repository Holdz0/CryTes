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

# TensorFlow log seviyesini ayarla
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

def ask_parent_followup(predicted_label, all_probs, classes, encoder):
    """
    Tespit edilen duruma göre ebeveyne takip soruları sorar ve öneride bulunur.
    """
    print("\n" + "="*50)
    print("📋 EBEVEYN TAKİP SORULARI")
    print("="*50)
    
    # En yüksek ikinci olasılığı bul (alternatif öneri için)
    probs_with_labels = [(classes[i], all_probs[i]*100) for i in range(len(classes))]
    probs_with_labels.sort(key=lambda x: x[1], reverse=True)
    second_best_label = probs_with_labels[1][0] if len(probs_with_labels) > 1 else None
    second_best_tr = LABEL_TR.get(second_best_label, second_best_label) if second_best_label else "Diğer"
    
    if predicted_label == "hungry":
        # AÇLIK TESPİTİ
        print("\n🍼 Açlık tespit edildi!")
        print("❓ Bebek son 2 saat içerisinde yemek yedi mi?")
        print("   [1] Evet")
        print("   [2] Hayır")
        
        while True:
            try:
                answer = input("\nCevabınızı girin (1 veya 2): ").strip()
                if answer == "1":
                    print(f"\n💡 ÖNERİ: Bebek yakın zamanda yemek yediği için, ağlamanın sebebi {second_best_tr} olabilir.")
                    print(f"   İkinci en yüksek tespit: {second_best_tr} (%{probs_with_labels[1][1]:.1f})")
                    break
                elif answer == "2":
                    print("\n🍼 SONUÇ: Bebeğiniz aç! Lütfen bebeğinizi besleyin.")
                    break
                else:
                    print("❌ Lütfen 1 veya 2 girin.")
            except ValueError:
                print("❌ Geçersiz giriş. Lütfen 1 veya 2 girin.")
    
    elif predicted_label == "discomfort":
        # RAHATSIZLIK TESPİTİ
        print("\n😫 Rahatsızlık/Huzursuzluk tespit edildi!")
        print("❓ Bebeğin altı son 4 saat içerisinde temizlendi mi?")
        print("   [1] Evet")
        print("   [2] Hayır")
        
        while True:
            try:
                answer = input("\nCevabınızı girin (1 veya 2): ").strip()
                if answer == "1":
                    print(f"\n💡 ÖNERİ: Bebeğin altı temiz olduğu için, ağlamanın sebebi {second_best_tr} olabilir.")
                    print(f"   İkinci en yüksek tespit: {second_best_tr} (%{probs_with_labels[1][1]:.1f})")
                    break
                elif answer == "2":
                    print("\n🧷 SONUÇ: Bebeğinizin altını temizlemeniz gerekiyor!")
                    break
                else:
                    print("❌ Lütfen 1 veya 2 girin.")
            except ValueError:
                print("❌ Geçersiz giriş. Lütfen 1 veya 2 girin.")
    
    elif predicted_label == "tired":
        # YORGUNLUK TESPİTİ
        print("\n😴 Yorgunluk tespit edildi!")
        print("❓ Bebek bugün toplam 12 saat uyudu mu?")
        print("   [1] Evet")
        print("   [2] Hayır")
        
        while True:
            try:
                answer = input("\nCevabınızı girin (1 veya 2): ").strip()
                if answer == "1":
                    print(f"\n💡 ÖNERİ: Bebek yeterli uyku almış görünüyor, ağlamanın sebebi {second_best_tr} olabilir.")
                    print(f"   İkinci en yüksek tespit: {second_best_tr} (%{probs_with_labels[1][1]:.1f})")
                    break
                elif answer == "2":
                    print("\n🛏️ SONUÇ: Bebeğinizin uyuması gerekiyor! Lütfen onu uyutmaya çalışın.")
                    break
                else:
                    print("❌ Lütfen 1 veya 2 girin.")
            except ValueError:
                print("❌ Geçersiz giriş. Lütfen 1 veya 2 girin.")
    
    elif predicted_label == "burping":
        # GAZ/GEĞİRME TESPİTİ
        print("\n💨 Gaz/Geğirme tespit edildi!")
        print("❓ Bebek gazını çıkarabildi mi?")
        print("   [1] Evet")
        print("   [2] Hayır")
        
        while True:
            try:
                answer = input("\nCevabınızı girin (1 veya 2): ").strip()
                if answer == "1":
                    print(f"\n💡 ÖNERİ: Bebek gazını çıkarmış görünüyor, ağlamanın sebebi {second_best_tr} olabilir.")
                    print(f"   İkinci en yüksek tespit: {second_best_tr} (%{probs_with_labels[1][1]:.1f})")
                    break
                elif answer == "2":
                    print("\n💨 SONUÇ: Bebeğinizin gazını çıkartması gerekiyor! Lütfen bebeğe gaz çıkartma egzersizleri yapın.")
                    break
                else:
                    print("❌ Lütfen 1 veya 2 girin.")
            except ValueError:
                print("❌ Geçersiz giriş. Lütfen 1 veya 2 girin.")
    
    elif predicted_label == "belly_pain":
        # KARIN AĞRISI TESPİTİ (Ek olarak ekledim)
        print("\n😣 Karın ağrısı tespit edildi!")
        print("❓ Bebek son öğünden sonra rahatsızlandı mı?")
        print("   [1] Evet")
        print("   [2] Hayır")
        
        while True:
            try:
                answer = input("\nCevabınızı girin (1 veya 2): ").strip()
                if answer == "1":
                    print("\n⚠️ SONUÇ: Bebek yemekten sonra rahatsızlanmış olabilir. Gaz veya hazımsızlık olabilir.")
                    print("   Bebeğin karnını hafifçe ovarak rahatlatmayı deneyin.")
                    break
                elif answer == "2":
                    print(f"\n💡 ÖNERİ: Karın ağrısının başka bir sebebi olabilir veya {second_best_tr} durumu söz konusu olabilir.")
                    print(f"   İkinci en yüksek tespit: {second_best_tr} (%{probs_with_labels[1][1]:.1f})")
                    break
                else:
                    print("❌ Lütfen 1 veya 2 girin.")
            except ValueError:
                print("❌ Geçersiz giriş. Lütfen 1 veya 2 girin.")
    
    else:
        print(f"\nℹ️ Tespit edilen durum: {LABEL_TR.get(predicted_label, predicted_label)}")
        print("   Bu durum için özel bir takip sorusu bulunmuyor.")
    
    print("\n" + "="*50)

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
    
    # Audio Buffer (Rolling Window) - 5 saniye
    BUFFER_SIZE = int(SAMPLE_RATE * DURATION)
    CHUNK_SIZE = int(SAMPLE_RATE * 0.5) # 0.5 saniyelik okumalar
    
    # Ring Buffer (Verimli)
    import collections
    audio_buffer = collections.deque(maxlen=BUFFER_SIZE)
    
    # YAMNet Sınıf İsimlerini Yükle (Yamnet modelinden)
    try:
        class_map_path = yamnet.class_map_path().numpy().decode('utf-8')
        class_names = [x['display_name'] for x in tf.io.read_file(class_map_path).numpy().decode('utf-8').splitlines()[1:] for x in [dict(zip(['index', 'mid', 'display_name'], x.split(',')))]]
    except:
        # Fallback (Standart YAMNet endeksleri)
        print("⚠️ YAMNet class map okunamadı, varsayılan endeksler kullanılıyor.")
        class_names = [] 
    
    print("\n" + "="*60)
    print(f"🎤 GELİŞMİŞ BEBEK AĞLAMASI ALGILAYICI (SMART LISTEN)")
    print(f"🧠 Mod: Sürekli Dinleme + Akıllı Tetikleme")
    print(f"⏱️  Tampon Bellek: {DURATION} sn")
    print("="*60 + "\n")

    print(f"👂 Dinleniyor... (Sessiz mod, ağlama bekleniyor)")
    
    last_log_time = time.time()
    
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE, device=device_index) as stream:
            while True:
                # 1. Chunk Oku
                chunk, _ = stream.read(CHUNK_SIZE)
                chunk = chunk.flatten()
                
                # Buffer'a ekle
                audio_buffer.extend(chunk)
                
                # Buffer dolmadan işlem yapma (ilk açılışta)
                if len(audio_buffer) < BUFFER_SIZE:
                    continue
                
                # 2. RMS (Enerji) Kontrolü - Hızlı Eleme
                # Son eklenen chunk'ın enerjisine bakıyoruz
                rms = np.sqrt(np.mean(np.array(chunk)**2))
                
                if rms < RMS_THRESHOLD:
                    # Sessiz, işlem yapma
                    # print(f"Wait... RMS: {rms:.4f}", end="\r") 
                    continue
                
                # Ses var! Şimdi YAMNet ile ne sesi olduğuna bakalım.
                # Buffer'ı numpy array'e çevir
                full_audio = np.array(audio_buffer)
                
                # Normalizasyon
                waveform = full_audio / np.max(np.abs(full_audio) + 1e-9)
                
                # 3. YAMNet "Gatekeeper" (Ön Eleme)
                # Sadece skorları alalım
                scores, embeddings, spectrogram = yamnet(waveform)
                
                # Skorların ortalamasını al (tüm klipler için)
                mean_scores = np.mean(scores, axis=0)
                
                is_baby_crying = False
                top3_indices = np.argsort(mean_scores)[::-1][:3]
                top_class_name = class_names[top3_indices[0]] if class_names else str(top3_indices[0])
                top_score = mean_scores[top3_indices[0]] * 100
                
                # 'Baby Cry' kontrolü (YAMNet Sınıf ID'leri: 20=Baby cry, 21=Crying, 22=Whimper)
                baby_indices = [20, 21, 22] 
                
                is_baby_crying = False
                detected_baby_score = 0.0
                
                # Top 3 yerine DOĞRUDAN bu indekslerin puanına bakıyoruz
                # Eğer herhangi biri > %5 - %10 ise tetikle
                for idx in baby_indices:
                    score = mean_scores[idx] * 100
                    if score > 5.0: # Çok hassas eşik (%5)
                        is_baby_crying = True
                        if score > detected_baby_score:
                            detected_baby_score = score
                
                current_time = time.time()
                
                # EŞİK KONTROLÜ: Score > 5.0 ise gir
                if is_baby_crying:
                    print(f"\n👶 BEBEK AĞLAMASI TESPİT EDİLDİ! (Puan: %{detected_baby_score:.1f})")
                    print(f"   (Algılanan: {class_names[top3_indices[0]] if class_names else top3_indices[0]})")
                    print("🔍 Sebebi analizi ediliyor...")
                    
                    # 4. Asıl Sınıflandırıcı (Transfer Learning)
                    global_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
                    
                    prediction = classifier.predict(global_embedding, verbose=0)[0]
                    predicted_idx = np.argmax(prediction)
                    confidence = prediction[predicted_idx] * 100
                    
                    if confidence < CONFIDENCE_THRESHOLD:
                        print(f"⚠️  Belirsiz Sonuç (%{confidence:.1f})")
                        print_prediction_bar(prediction, classes, predicted_idx)
                    else:
                        predicted_label = encoder.inverse_transform([predicted_idx])[0]
                        tr_label = LABEL_TR.get(predicted_label, predicted_label)
                        
                        print(f"🎯 SONUÇ: {tr_label}")
                        print(f"✅ Güven: %{confidence:.1f}")
                        print_prediction_bar(prediction, classes, predicted_idx)
                        
                        # Arduino'ya gönder
                        send_to_arduino(arduino, tr_label, confidence)
                        
                        # Ebeveyne takip soruları sor
                        ask_parent_followup(predicted_label, prediction, classes, encoder)
                    
                    print("-" * 50)
                    print("💤 3 saniye bekleme...")
                    time.sleep(3)
                    audio_buffer.clear()
                    print("👂 Dinlemeye devam ediliyor...")
                
                else:
                    # Bebek ağlaması YOKSA
                    # Her 2.5 saniyede bir log bas (Sıklığı artırdım)
                    if current_time - last_log_time > 2.5:
                        print(f"🔉 Ses Var: {top_class_name} (%{top_score:.1f}) - Bebek Sesi Yok (<%5) ❌")
                        last_log_time = current_time
                 
    except Exception as e:
        print(f"\n❌ Beklenmeyen Hata: {e}")
        import traceback
        traceback.print_exc()

    except KeyboardInterrupt:
        print("\n🛑 Çıkış.")

if __name__ == "__main__":
    main()
