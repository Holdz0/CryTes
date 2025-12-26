package com.ciona.babycry.ml

import android.content.Context
import android.util.Log
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
        private const val TAG = "CryClassifier"
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
    private var inputShape: IntArray? = null
    private var outputShape: IntArray? = null
    
    init {
        try {
            val model = loadModelFile(context, MODEL_FILE)
            val options = Interpreter.Options().apply {
                setNumThreads(2)
            }
            interpreter = Interpreter(model, options)
            
            // Model yapısını analiz et
            analyzeModel()
            
            Log.d(TAG, "CryClassifier model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load CryClassifier model", e)
            throw e
        }
    }
    
    private fun analyzeModel() {
        val interp = interpreter ?: return
        
        inputShape = interp.getInputTensor(0).shape()
        outputShape = interp.getOutputTensor(0).shape()
        
        Log.d(TAG, "Input shape: ${inputShape?.contentToString()}")
        Log.d(TAG, "Output shape: ${outputShape?.contentToString()}")
    }
    
    /**
     * Embedding'den sebep sınıflandırması yap
     */
    fun classify(embedding: FloatArray): ClassificationResult {
        val interp = interpreter ?: return createDefaultResult()
        
        try {
            // Embedding boyutunu kontrol et
            val actualInputSize = inputShape?.lastOrNull() ?: EMBEDDING_SIZE
            
            // Input tensor hazırla
            val inputBuffer = ByteBuffer.allocateDirect(actualInputSize * 4)
                .order(ByteOrder.nativeOrder())
            
            for (i in 0 until actualInputSize) {
                val value = if (i < embedding.size) embedding[i] else 0f
                inputBuffer.putFloat(value)
            }
            inputBuffer.rewind()
            
            // Output tensor hazırla
            val outputSize = outputShape?.lastOrNull() ?: NUM_CLASSES
            val outputBuffer = ByteBuffer.allocateDirect(outputSize * 4)
                .order(ByteOrder.nativeOrder())
            
            // Model çalıştır
            interp.run(inputBuffer, outputBuffer)
            
            // Sonuçları oku
            outputBuffer.rewind()
            val probabilities = FloatArray(outputSize)
            for (i in 0 until outputSize) {
                probabilities[i] = outputBuffer.float
            }
            
            // Softmax uygula (eğer gerekiyorsa)
            val normalizedProbs = softmax(probabilities)
            
            // En yüksek olasılıklı sınıfı bul
            var maxIdx = 0
            var maxProb = normalizedProbs[0]
            for (i in 1 until minOf(normalizedProbs.size, NUM_CLASSES)) {
                if (normalizedProbs[i] > maxProb) {
                    maxProb = normalizedProbs[i]
                    maxIdx = i
                }
            }
            
            val predictedClass = if (maxIdx < CLASS_NAMES.size) CLASS_NAMES[maxIdx] else "unknown"
            
            Log.d(TAG, "Predicted: $predictedClass with ${maxProb * 100}% confidence")
            
            return ClassificationResult(
                predictedClass = predictedClass,
                predictedLabel = CLASS_LABELS_TR[predictedClass] ?: predictedClass,
                emoji = CLASS_EMOJIS[predictedClass] ?: "❓",
                confidence = maxProb,
                isConfident = maxProb >= CONFIDENCE_THRESHOLD,
                allProbabilities = CLASS_NAMES.take(minOf(CLASS_NAMES.size, normalizedProbs.size))
                    .zip(normalizedProbs.toList()).toMap()
            )
            
        } catch (e: Exception) {
            Log.e(TAG, "Error during classification", e)
            return createDefaultResult()
        }
    }
    
    private fun softmax(logits: FloatArray): FloatArray {
        val maxLogit = logits.maxOrNull() ?: 0f
        val expSum = logits.sumOf { kotlin.math.exp((it - maxLogit).toDouble()) }
        return FloatArray(logits.size) { 
            (kotlin.math.exp((logits[it] - maxLogit).toDouble()) / expSum).toFloat()
        }
    }
    
    private fun createDefaultResult(): ClassificationResult {
        return ClassificationResult(
            predictedClass = "unknown",
            predictedLabel = "Bilinmiyor",
            emoji = "❓",
            confidence = 0f,
            isConfident = false,
            allProbabilities = emptyMap()
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
    val predictedClass: String,
    val predictedLabel: String,
    val emoji: String,
    val confidence: Float,
    val isConfident: Boolean,
    val allProbabilities: Map<String, Float>
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
