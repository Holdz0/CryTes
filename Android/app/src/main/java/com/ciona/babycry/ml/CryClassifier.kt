package com.ciona.babycry.ml

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * CryClassifier - Ağlama sebebi sınıflandırıcısı
 * 
 * YAMNet embedding'ini alır ve 5 sınıftan birine sınıflandırır:
 * - hungry (Açlık)
 * - belly_pain (Karın Ağrısı)
 * - burping (Gaz/Geğirme)
 * - discomfort (Rahatsızlık)
 * - tired (Yorgunluk)
 */
class CryClassifier(context: Context) {
    
    companion object {
        private const val MODEL_FILE = "cry_classifier.tflite"
        private const val EMBEDDING_SIZE = 1024
        private const val NUM_CLASSES = 5
        const val CONFIDENCE_THRESHOLD = 0.40f  // %40 güven eşiği
        
        // Sınıf isimleri (model eğitimindeki sıraya göre)
        val CLASS_NAMES = arrayOf(
            "belly_pain",
            "burping", 
            "discomfort",
            "hungry",
            "tired"
        )
        
        // Türkçe etiketler
        val CLASS_LABELS_TR = mapOf(
            "hungry" to "Açlık",
            "belly_pain" to "Karın Ağrısı",
            "burping" to "Gaz/Geğirme",
            "discomfort" to "Rahatsızlık",
            "tired" to "Yorgunluk"
        )
        
        // Emoji'ler
        val CLASS_EMOJIS = mapOf(
            "hungry" to "🍼",
            "belly_pain" to "😣",
            "burping" to "💨",
            "discomfort" to "😫",
            "tired" to "😴"
        )
    }
    
    private var interpreter: Interpreter? = null
    
    init {
        val model = loadModelFile(context, MODEL_FILE)
        interpreter = Interpreter(model)
    }
    
    /**
     * Embedding'den sebep sınıflandırması yap
     * 
     * @param embedding 1024 boyutlu YAMNet özellik vektörü
     * @return ClassificationResult tüm sınıf olasılıkları ile
     */
    fun classify(embedding: FloatArray): ClassificationResult {
        val interpreter = this.interpreter ?: throw IllegalStateException("Model not loaded")
        
        require(embedding.size == EMBEDDING_SIZE) {
            "Embedding size must be $EMBEDDING_SIZE, got ${embedding.size}"
        }
        
        // Input tensor hazırla
        val inputBuffer = ByteBuffer.allocateDirect(EMBEDDING_SIZE * 4)
            .order(ByteOrder.nativeOrder())
        for (value in embedding) {
            inputBuffer.putFloat(value)
        }
        inputBuffer.rewind()
        
        // Output tensor hazırla
        val outputBuffer = ByteBuffer.allocateDirect(NUM_CLASSES * 4)
            .order(ByteOrder.nativeOrder())
        
        // Model çalıştır
        interpreter.run(inputBuffer, outputBuffer)
        
        // Sonuçları oku
        outputBuffer.rewind()
        val probabilities = FloatArray(NUM_CLASSES)
        for (i in 0 until NUM_CLASSES) {
            probabilities[i] = outputBuffer.float
        }
        
        // En yüksek olasılıklı sınıfı bul
        var maxIdx = 0
        var maxProb = probabilities[0]
        for (i in 1 until NUM_CLASSES) {
            if (probabilities[i] > maxProb) {
                maxProb = probabilities[i]
                maxIdx = i
            }
        }
        
        val predictedClass = CLASS_NAMES[maxIdx]
        
        return ClassificationResult(
            predictedClass = predictedClass,
            predictedLabel = CLASS_LABELS_TR[predictedClass] ?: predictedClass,
            emoji = CLASS_EMOJIS[predictedClass] ?: "❓",
            confidence = maxProb,
            isConfident = maxProb >= CONFIDENCE_THRESHOLD,
            allProbabilities = CLASS_NAMES.zip(probabilities.toList()).toMap()
        )
    }
    
    /**
     * TFLite model dosyasını yükle
     */
    private fun loadModelFile(context: Context, filename: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    /**
     * Kaynakları serbest bırak
     */
    fun close() {
        interpreter?.close()
        interpreter = null
    }
}

/**
 * Sınıflandırma sonucu
 */
data class ClassificationResult(
    val predictedClass: String,              // Tahmin edilen sınıf (İngilizce)
    val predictedLabel: String,              // Türkçe etiket
    val emoji: String,                       // İlgili emoji
    val confidence: Float,                   // Güven oranı (0-1)
    val isConfident: Boolean,                // Eşiği geçti mi?
    val allProbabilities: Map<String, Float> // Tüm sınıf olasılıkları
) {
    /**
     * LCD için ASCII temizlenmiş metin
     */
    fun getLcdText(): String {
        val cleanLabel = predictedLabel
            .replace("ı", "i")
            .replace("ğ", "g")
            .replace("ü", "u")
            .replace("ş", "s")
            .replace("ö", "o")
            .replace("ç", "c")
            .replace("İ", "I")
            .replace("Ğ", "G")
            .replace("Ü", "U")
            .replace("Ş", "S")
            .replace("Ö", "O")
            .replace("Ç", "C")
        
        return "${cleanLabel.take(16)}%${(confidence * 100).toInt()} Guven"
    }
}
