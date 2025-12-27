# -*- coding: utf-8 -*-
"""
Live Detection Script (YAMNet Transfer Learning Version)
Birleştirilmiş Versiyon: LCD + Sensör + Ebeveyn Takip Soruları
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
CONFIDENCE_THRESHOLD = 40.0
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
        print("  - YAMNet indiriliyor/yükleniyor...")
        yamnet = hub.load(YAMNET_MODEL_HANDLE)
        
        print("  - Sınıflandırıcı yükleniyor...")
        classifier = tf.keras.models.load_model(MODEL_PATH)
        
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
    waveform = audio_data / np.max(np.abs(audio_data) + 1e-9)
    _, embeddings, _ = yamnet(waveform)
    global_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
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
        time.sleep(2)
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
        lcd_text = label.replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
        lcd_text = lcd_text.replace("İ", "I").replace("Ğ", "G").replace("Ü", "U").replace("Ş", "S").replace("Ö", "O").replace("Ç", "C")
        for emoji in ['🍼', '😣', '💨', '😫', '😴']:
            lcd_text = lcd_text.replace(emoji, '')
        lcd_text = lcd_text.strip()
        
        message = f"{lcd_text[:16]}%{confidence:.0f} Guven"
        arduino.write(f"{message}\n".encode('ascii', errors='ignore'))
        print(f"📟 LCD'ye gönderildi: {lcd_text}")
    except Exception as e:
        print(f"⚠️ Arduino gönderim hatası: {e}")

def send_status_to_arduino(arduino, line1, line2="", scroll=False, display_time=0):
    """LCD'ye durum mesajı gönder (üst satır, alt satır)
    scroll=True ise uzun yazılar kaydırılır
    display_time>0 ise o kadar saniye ekranda kalır
    """
    if arduino is None:
        return
    
    def fix_turkish(text):
        text = text.replace("ı", "i").replace("ğ", "g").replace("ü", "u").replace("ş", "s").replace("ö", "o").replace("ç", "c")
        text = text.replace("İ", "I").replace("Ğ", "G").replace("Ü", "U").replace("Ş", "S").replace("Ö", "O").replace("Ç", "C")
        for emoji in ['🍼', '😣', '💨', '😫', '😴', '👂', '🔉', '❌', '✅', '🎯', '👶']:
            text = text.replace(emoji, '')
        return text.strip()
    
    try:
        l1 = fix_turkish(line1)
        l2 = fix_turkish(line2)
        
        if scroll and (len(l1) > 10 or len(l2) > 10):
            # Kayan yazı modu - döngü halinde
            l1_padded = "   " + l1 + "   " if len(l1) > 10 else l1.center(16)
            l2_padded = "   " + l2 + "   " if len(l2) > 10 else l2.center(16)
            
            scroll_speed = 0.3
            start_time = time.time()
            total_time = display_time if display_time > 0 else 3
            
            while (time.time() - start_time) < total_time:
                max_steps = max(len(l1_padded), len(l2_padded)) - 15
                for i in range(max(1, max_steps)):
                    if (time.time() - start_time) >= total_time:
                        break
                    s1 = l1_padded[i:i+16] if len(l1_padded) > 16 else l1_padded[:16]
                    s2 = l2_padded[i:i+16] if len(l2_padded) > 16 else l2_padded[:16]
                    message = f"{s1}%{s2}"
                    arduino.write(f"{message}\n".encode('ascii', errors='ignore'))
                    time.sleep(scroll_speed)
        else:
            message = f"{l1[:16]}%{l2[:16]}"
            arduino.write(f"{message}\n".encode('ascii', errors='ignore'))
            if display_time > 0:
                time.sleep(display_time)
                
    except Exception as e:
        print(f"⚠️ Arduino durum gönderim hatası: {e}")

def read_sensor_data(arduino):
    """Arduino'dan sensör verisi oku"""
    if arduino is None:
        return None, None
    try:
        arduino.write(b"GET_SENSOR\n")
        time.sleep(0.3)
        
        for _ in range(5):
            if arduino.in_waiting > 0:
                line = arduino.readline().decode('ascii', errors='ignore').strip()
                print(f"   [DEBUG] Arduino: {line}")
                if line.startswith("SENSOR:"):
                    data = line.replace("SENSOR:", "").split(",")
                    if len(data) == 2:
                        temp = float(data[0])
                        hum = float(data[1])
                        return temp, hum
            time.sleep(0.1)
    except Exception as e:
        print(f"   [DEBUG] Sensör okuma hatası: {e}")
    return None, None

def set_traffic_light(arduino, state):
    """
    Trafik lambasını kontrol et (Pin 3=Yeşil, Pin 4=Sarı, Pin 5=Kırmızı)
    state: 'GREEN' (sessiz), 'YELLOW' (ağlama harici ses), 'RED' (bebek ağlıyor)
    """
    if arduino is None:
        return
    try:
        arduino.write(f"LIGHT:{state}\n".encode('ascii', errors='ignore'))
        state_tr = {'GREEN': '🟢 Yeşil (Sessiz)', 'YELLOW': '🟡 Sarı (Ses Var)', 'RED': '🔴 Kırmızı (Ağlama)'}
        print(f"🚦 Trafik Lambası: {state_tr.get(state, state)}")
    except Exception as e:
        print(f"⚠️ Trafik lambası hatası: {e}")

def play_lullaby(arduino):
    """
    Arduino'ya ninni çalma komutu gönder
    Buzzer ile Dandini Dandini Dastana çalar, LED'ler sırayla yanar
    """
    if arduino is None:
        return
    try:
        print("🎵 Ninni başlatılıyor (Dandini Dandini Dastana)...")
        arduino.write(b"PLAY_LULLABY\n")
        # Ninni yaklaşık 25-30 saniye sürer, o kadar bekle
        print("   💤 Ninni çalıyor... (Lütfen bekleyin)")
    except Exception as e:
        print(f"⚠️ Ninni başlatma hatası: {e}")

def check_environment(temp, hum):
    """Ortam koşullarını kontrol et ve uyarı mesajı döndür"""
    warnings = []
    lcd_warnings = []
    
    TEMP_HIGH = 28.0
    TEMP_LOW = 18.0
    HUM_HIGH = 70.0
    HUM_LOW = 30.0
    
    if temp is not None:
        if temp > TEMP_HIGH:
            warnings.append(f"🌡️ Sıcak! ({temp:.1f}°C) - Bebek terliyor olabilir")
            lcd_warnings.append(("Terliyor Olabilir", f"Sicak {temp:.0f}C"))
        elif temp < TEMP_LOW:
            warnings.append(f"❄️ Soğuk! ({temp:.1f}°C) - Bebek üşüyor olabilir")
            lcd_warnings.append(("Usuyor Olabilir", f"Soguk {temp:.0f}C"))
    
    if hum is not None:
        if hum > HUM_HIGH:
            warnings.append(f"💧 Nem yüksek! (%{hum:.0f}) - Bunaltıcı olabilir")
            lcd_warnings.append(("Terliyor Olabilir", f"Nem Yuksek %{hum:.0f}"))
        elif hum < HUM_LOW:
            warnings.append(f"🏜️ Nem düşük! (%{hum:.0f}) - Hava kuru")
            lcd_warnings.append(("Kuru Hava Uyarisi", f"Nem Dusuk %{hum:.0f}"))
    
    return warnings, lcd_warnings

def ask_parent_followup(predicted_label, all_probs, classes, encoder):
    """Tespit edilen duruma göre ebeveyne takip soruları sorar ve öneride bulunur."""
    print("\n" + "="*50)
    print("📋 EBEVEYN TAKİP SORULARI")
    print("="*50)
    
    probs_with_labels = [(classes[i], all_probs[i]*100) for i in range(len(classes))]
    probs_with_labels.sort(key=lambda x: x[1], reverse=True)
    second_best_label = probs_with_labels[1][0] if len(probs_with_labels) > 1 else None
    second_best_tr = LABEL_TR.get(second_best_label, second_best_label) if second_best_label else "Diğer"
    
    if predicted_label == "hungry":
        print("\n🍼 Açlık tespit edildi!")
        print("❓ Bebek son 2 saat içerisinde yemek yedi mi?")
        print("   [1] Evet")
        print("   [2] Hayır")
        
        while True:
            try:
                answer = input("\nCevabınızı girin (1 veya 2): ").strip()
                if answer == "1":
                    print(f"\n💡 ÖNERİ: Bebek yakın zamanda yemek yediği için, ağlamanın sebebi {second_best_tr} olabilir.")
                    break
                elif answer == "2":
                    print("\n🍼 SONUÇ: Bebeğiniz aç! Lütfen bebeğinizi besleyin.")
                    break
                else:
                    print("❌ Lütfen 1 veya 2 girin.")
            except ValueError:
                print("❌ Geçersiz giriş.")
    
    elif predicted_label == "discomfort":
        print("\n😫 Rahatsızlık/Huzursuzluk tespit edildi!")
        print("❓ Bebeğin altı son 4 saat içerisinde temizlendi mi?")
        print("   [1] Evet")
        print("   [2] Hayır")
        
        while True:
            try:
                answer = input("\nCevabınızı girin (1 veya 2): ").strip()
                if answer == "1":
                    print(f"\n💡 ÖNERİ: Bebeğin altı temiz olduğu için, ağlamanın sebebi {second_best_tr} olabilir.")
                    break
                elif answer == "2":
                    print("\n🧷 SONUÇ: Bebeğinizin altını temizlemeniz gerekiyor!")
                    break
                else:
                    print("❌ Lütfen 1 veya 2 girin.")
            except ValueError:
                print("❌ Geçersiz giriş.")
    
    elif predicted_label == "tired":
        print("\n😴 Yorgunluk tespit edildi!")
        print("❓ Bebek bugün toplam 12 saat uyudu mu?")
        print("   [1] Evet")
        print("   [2] Hayır")
        
        while True:
            try:
                answer = input("\nCevabınızı girin (1 veya 2): ").strip()
                if answer == "1":
                    print(f"\n💡 ÖNERİ: Bebek yeterli uyku almış görünüyor, ağlamanın sebebi {second_best_tr} olabilir.")
                    break
                elif answer == "2":
                    print("\n🛏️ SONUÇ: Bebeğinizin uyuması gerekiyor!")
                    break
                else:
                    print("❌ Lütfen 1 veya 2 girin.")
            except ValueError:
                print("❌ Geçersiz giriş.")
    
    elif predicted_label == "burping":
        print("\n💨 Gaz/Geğirme tespit edildi!")
        print("❓ Bebek gazını çıkarabildi mi?")
        print("   [1] Evet")
        print("   [2] Hayır")
        
        while True:
            try:
                answer = input("\nCevabınızı girin (1 veya 2): ").strip()
                if answer == "1":
                    print(f"\n💡 ÖNERİ: Bebek gazını çıkarmış görünüyor, ağlamanın sebebi {second_best_tr} olabilir.")
                    break
                elif answer == "2":
                    print("\n💨 SONUÇ: Bebeğinizin gazını çıkartması gerekiyor!")
                    break
                else:
                    print("❌ Lütfen 1 veya 2 girin.")
            except ValueError:
                print("❌ Geçersiz giriş.")
    
    elif predicted_label == "belly_pain":
        print("\n😣 Karın ağrısı tespit edildi!")
        print("❓ Bebek son öğünden sonra rahatsızlandı mı?")
        print("   [1] Evet")
        print("   [2] Hayır")
        
        while True:
            try:
                answer = input("\nCevabınızı girin (1 veya 2): ").strip()
                if answer == "1":
                    print("\n⚠️ SONUÇ: Bebek yemekten sonra rahatsızlanmış olabilir. Gaz veya hazımsızlık olabilir.")
                    break
                elif answer == "2":
                    print(f"\n💡 ÖNERİ: Karın ağrısının başka bir sebebi olabilir veya {second_best_tr} durumu söz konusu olabilir.")
                    break
                else:
                    print("❌ Lütfen 1 veya 2 girin.")
            except ValueError:
                print("❌ Geçersiz giriş.")
    
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
    print(f"🎯  Güven Eşiği: %{CONFIDENCE_THRESHOLD}")
    print("="*60 + "\n")
    
    BUFFER_SIZE = int(SAMPLE_RATE * DURATION)
    CHUNK_SIZE = int(SAMPLE_RATE * 0.5)
    
    import collections
    audio_buffer = collections.deque(maxlen=BUFFER_SIZE)
    
    try:
        class_map_path = yamnet.class_map_path().numpy().decode('utf-8')
        class_names = [x['display_name'] for x in tf.io.read_file(class_map_path).numpy().decode('utf-8').splitlines()[1:] for x in [dict(zip(['index', 'mid', 'display_name'], x.split(',')))]]
    except:
        print("⚠️ YAMNet class map okunamadı, varsayılan endeksler kullanılıyor.")
        class_names = [] 
    
    print("\n" + "="*60)
    print(f"🎤 GELİŞMİŞ BEBEK AĞLAMASI ALGILAYICI (SMART LISTEN)")
    print(f"🧠 Mod: Sürekli Dinleme + Akıllı Tetikleme")
    print(f"⏱️  Tampon Bellek: {DURATION} sn")
    print("="*60 + "\n")

    print(f"👂 Dinleniyor... (Sessiz mod, ağlama bekleniyor)")
    
    send_status_to_arduino(arduino, "Dinleniyor...", "Bebek bekleniyor")
    set_traffic_light(arduino, 'GREEN')  # Başlangıçta yeşil - sessiz
    
    last_log_time = time.time()
    
    try:
        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE, device=device_index) as stream:
            while True:
                chunk, _ = stream.read(CHUNK_SIZE)
                chunk = chunk.flatten()
                
                audio_buffer.extend(chunk)
                
                if len(audio_buffer) < BUFFER_SIZE:
                    continue
                
                rms = np.sqrt(np.mean(np.array(chunk)**2))
                
                if rms < RMS_THRESHOLD:
                    set_traffic_light(arduino, 'GREEN')  # Sessiz - yeşil
                    continue
                
                full_audio = np.array(audio_buffer)
                waveform = full_audio / np.max(np.abs(full_audio) + 1e-9)
                
                scores, embeddings, spectrogram = yamnet(waveform)
                mean_scores = np.mean(scores, axis=0)
                
                is_baby_crying = False
                top3_indices = np.argsort(mean_scores)[::-1][:3]
                top_class_name = class_names[top3_indices[0]] if class_names else str(top3_indices[0])
                top_score = mean_scores[top3_indices[0]] * 100
                
                baby_indices = [20, 21, 22] 
                
                is_baby_crying = False
                detected_baby_score = 0.0
                
                for idx in baby_indices:
                    score = mean_scores[idx] * 100
                    if score > 5.0:
                        is_baby_crying = True
                        if score > detected_baby_score:
                            detected_baby_score = score
                
                current_time = time.time()
                
                if is_baby_crying:
                    set_traffic_light(arduino, 'RED')  # Bebek ağlıyor - kırmızı
                    print(f"\n👶 BEBEK AĞLAMASI TESPİT EDİLDİ! (Puan: %{detected_baby_score:.1f})")
                    print(f"   (Algılanan: {class_names[top3_indices[0]] if class_names else top3_indices[0]})")
                    print("🔍 Sebebi analizi ediliyor...")
                    
                    global_embedding = np.mean(embeddings, axis=0).reshape(1, -1)
                    
                    prediction = classifier.predict(global_embedding, verbose=0)[0]
                    predicted_idx = np.argmax(prediction)
                    confidence = prediction[predicted_idx] * 100
                    
                    # Her zaman en yüksek sonucu göster ve Arduino'ya gönder
                    predicted_label = encoder.inverse_transform([predicted_idx])[0]
                    tr_label = LABEL_TR.get(predicted_label, predicted_label)
                    
                    if confidence < CONFIDENCE_THRESHOLD:
                        print(f"⚠️  Düşük Güven (%{confidence:.1f}) - Yine de en yüksek sonuç gösteriliyor")
                    else:
                        print(f"✅ Güven: %{confidence:.1f}")
                    
                    print(f"🎯 SONUÇ: {tr_label}")
                    print_prediction_bar(prediction, classes, predicted_idx)
                    
                    # Arduino'ya gönder (güven düşük olsa bile)
                    send_to_arduino(arduino, tr_label, confidence)
                    
                    # Ortam kontrolü (Sensör verisi oku)
                    time.sleep(0.5)
                    temp, hum = read_sensor_data(arduino)
                    if temp is not None and hum is not None:
                        print(f"\n🌡️ Ortam: {temp:.1f}°C | 💧 Nem: %{hum:.0f}")
                        env_warnings, lcd_warnings = check_environment(temp, hum)
                        for i, warn in enumerate(env_warnings):
                            print(f"   ⚠️ {warn}")
                            if i < len(lcd_warnings):
                                line1, line2 = lcd_warnings[i]
                                send_status_to_arduino(arduino, line1, line2, scroll=True, display_time=5)
                    
                    # Yorgunluk veya Rahatsızlık ise ninni çal
                    if predicted_label in ['tired', 'discomfort']:
                        print("\n🌙 Bebek yorgun/rahatsız - Ninni başlatılıyor...")
                        send_status_to_arduino(arduino, "Ninni Caliyor", "Dandini Dastana")
                        play_lullaby(arduino)
                        # Ninni süresince bekle (yaklaşık 30 saniye)
                        time.sleep(30)
                        print("🎵 Ninni tamamlandı.")
                    
                    # Ebeveyne takip soruları sor
                    ask_parent_followup(predicted_label, prediction, classes, encoder)
                    
                    print("-" * 50)
                    print("💤 3 saniye bekleme...")
                    time.sleep(3)
                    audio_buffer.clear()
                    print("👂 Dinlemeye devam ediliyor...")
                    send_status_to_arduino(arduino, "Dinleniyor...", "Bebek bekleniyor")
                
                else:
                    set_traffic_light(arduino, 'YELLOW')  # Ağlama harici ses - sarı
                    if current_time - last_log_time > 2.5:
                        print(f"🔉 Ses Var: {top_class_name} (%{top_score:.1f}) - Bebek Sesi Yok (<%5) ❌")
                        send_status_to_arduino(arduino, f"Ses: {top_class_name[:10]}", f"%{top_score:.0f} - Bebek yok")
                        last_log_time = current_time
                 
    except Exception as e:
        print(f"\n❌ Beklenmeyen Hata: {e}")
        import traceback
        traceback.print_exc()

    except KeyboardInterrupt:
        print("\n🛑 Çıkış.")

if __name__ == "__main__":
    main()
