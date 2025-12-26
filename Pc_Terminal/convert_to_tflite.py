# -*- coding: utf-8 -*-
"""
Model Dönüştürücü: Keras H5 → TensorFlow Lite
Crytes Android Uygulaması için
"""

import os
import numpy as np
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
H5_MODEL_PATH = os.path.join(SCRIPT_DIR, "yamnet_transfer_model.h5")
TFLITE_OUTPUT_PATH = os.path.join(SCRIPT_DIR, "cry_classifier.tflite")

def convert_h5_to_tflite():
    """Keras H5 modelini TFLite formatına dönüştür"""
    
    print("=" * 50)
    print("🔄 Model Dönüşümü Başlıyor")
    print("=" * 50)
    
    # 1. Keras modelini yükle
    print("\n1️⃣ Keras modeli yükleniyor...")
    model = tf.keras.models.load_model(H5_MODEL_PATH)
    model.summary()
    
    # 2. TFLite Converter oluştur
    print("\n2️⃣ TFLite'a dönüştürülüyor...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Float32 hassasiyet (accuracy kaybı yok)
    # Opsiyonel: Float16 için aşağıdaki satırları açabilirsin
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_types = [tf.float16]
    
    # Dönüştür
    tflite_model = converter.convert()
    
    # 3. Dosyaya kaydet
    print("\n3️⃣ Dosyaya kaydediliyor...")
    with open(TFLITE_OUTPUT_PATH, 'wb') as f:
        f.write(tflite_model)
    
    # Boyut bilgisi
    original_size = os.path.getsize(H5_MODEL_PATH) / (1024 * 1024)
    tflite_size = os.path.getsize(TFLITE_OUTPUT_PATH) / (1024 * 1024)
    
    print("\n" + "=" * 50)
    print("✅ DÖNÜŞÜM TAMAMLANDI!")
    print("=" * 50)
    print(f"📁 Orijinal (H5): {original_size:.2f} MB")
    print(f"📁 TFLite:        {tflite_size:.2f} MB")
    print(f"📍 Çıktı: {TFLITE_OUTPUT_PATH}")
    
    # 4. Doğrulama testi
    print("\n4️⃣ Doğrulama testi yapılıyor...")
    verify_tflite_model()
    
def verify_tflite_model():
    """TFLite modelini test et"""
    
    # TFLite Interpreter yükle
    interpreter = tf.lite.Interpreter(model_path=TFLITE_OUTPUT_PATH)
    interpreter.allocate_tensors()
    
    # Input/Output detayları
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"   Input shape:  {input_details[0]['shape']}")
    print(f"   Input dtype:  {input_details[0]['dtype']}")
    print(f"   Output shape: {output_details[0]['shape']}")
    print(f"   Output dtype: {output_details[0]['dtype']}")
    
    # Test verisi ile çalıştır
    test_input = np.random.randn(1, 1024).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    
    print(f"\n   🧪 Test sonucu (rastgele input):")
    print(f"   Output: {output}")
    print(f"   Sum: {np.sum(output):.4f} (softmax için ~1.0 olmalı)")
    
    if abs(np.sum(output) - 1.0) < 0.01:
        print("\n   ✅ Model doğrulaması BAŞARILI!")
    else:
        print("\n   ⚠️  Model çıktısı beklenenden farklı!")

if __name__ == "__main__":
    convert_h5_to_tflite()
