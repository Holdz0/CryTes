# Crytes - Bebek Ağlama Dedektörü (Android)

PC'deki Python bebek ağlaması algılama sisteminin Android uygulaması.

## 📱 Özellikler

- **Gerçek zamanlı ses algılama** (16kHz mono)
- **YAMNet ile ön eleme** (bebek ağlaması tespiti)
- **Transfer Learning sınıflandırması** (5 sebep)
- **Arduino USB OTG desteği** (LCD'ye sonuç gönderme)
- **Offline çalışma** (internet bağlantısı gerektirmez)

## 🏗️ Proje Yapısı

```
Android/
├── app/src/main/
│   ├── java/com/ciona/babycry/
│   │   ├── MainActivity.kt           # Ana UI
│   │   ├── audio/
│   │   │   └── AudioCapture.kt       # Mikrofon kaydı
│   │   ├── ml/
│   │   │   ├── YamnetProcessor.kt    # YAMNet embedding
│   │   │   └── CryClassifier.kt      # Sebep sınıflandırma
│   │   └── serial/
│   │       └── ArduinoSerial.kt      # USB OTG iletişim
│   ├── res/
│   │   ├── layout/activity_main.xml
│   │   └── values/
│   └── assets/
│       ├── yamnet.tflite             # YAMNet modeli (~3.2 MB)
│       └── cry_classifier.tflite     # Bizim modelimiz (~2.5 MB)
├── build.gradle.kts
└── settings.gradle.kts
```

## 🚀 Kurulum

### Gereksinimler

- Android Studio Arctic Fox veya üzeri
- Android SDK 34
- Kotlin 1.9+
- Android cihaz (API 24+, Android 7.0+)
- USB OTG kablosu (Arduino bağlantısı için)

### Derleme Adımları

1. **Android Studio'da Aç**
   - File → Open → `Android` klasörünü seç

2. **Gradle Sync**
   - Android Studio otomatik yapacak
   - İlk seferde biraz bekleyebilir

3. **Build**
   - Build → Make Project
   - Veya: `./gradlew assembleDebug`

4. **Çalıştır**
   - Run → Run 'app'
   - Emülatör veya gerçek cihaz seç

## 📋 Sınıflar

| Sınıf | Türkçe | Emoji |
|-------|--------|-------|
| hungry | Açlık | 🍼 |
| belly_pain | Karın Ağrısı | 😣 |
| burping | Gaz/Geğirme | 💨 |
| discomfort | Rahatsızlık | 😫 |
| tired | Yorgunluk | 😴 |

## 🔌 Arduino Bağlantısı

1. USB OTG kablosu ile Arduino'yu telefona bağla
2. Uygulama otomatik algılayacak
3. Sonuçlar LCD'ye gönderilecek

### Mesaj Formatı
```
{sebep}%{guven} Guven
```
Örnek: `Aclik%87 Guven`

## ⚙️ Parametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| SAMPLE_RATE | 16000 Hz | YAMNet zorunluluğu |
| BUFFER_DURATION | 5 saniye | Analiz penceresi |
| RMS_THRESHOLD | 0.005 | Sessizlik eşiği |
| CONFIDENCE_THRESHOLD | 40% | Güven eşiği |
| BABY_CRY_THRESHOLD | 5% | YAMNet bebek ağlaması eşiği |

## 📄 Lisans

Bu proje Hackathon Ciona için geliştirilmiştir.
