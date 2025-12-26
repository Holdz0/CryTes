package com.ciona.babycry.ml

import android.content.Context
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * YamnetProcessor - YAMNet TFLite ile ses embedding'i çıkarma
 * 
 * YAMNet modeli:
 * - Input: [1, 15600] float32 (0.975 saniye @ 16kHz)
 * - Output scores: [1, 521] (AudioSet sınıfları)
 * - Output embeddings: [1, 1024] (bizim kullanacağımız)
 */
class YamnetProcessor(context: Context) {
    
    companion object {
        private const val MODEL_FILE = "yamnet.tflite"
        private const val YAMNET_SAMPLE_LENGTH = 15600  // 0.975 saniye @ 16kHz
        private const val EMBEDDING_SIZE = 1024
        private const val NUM_CLASSES = 521
        
        // Baby cry sınıf indeksleri (YAMNet AudioSet)
        val BABY_CRY_INDICES = intArrayOf(20, 21, 22)  // Baby cry, infant cry, crying
        const val BABY_CRY_THRESHOLD = 0.05f  // %5 eşik
    }
    
    private var interpreter: Interpreter? = null
    
    init {
        val model = loadModelFile(context, MODEL_FILE)
        interpreter = Interpreter(model)
    }
    
    /**
     * Ses verisinden embedding ve baby cry skoru çıkar
     * 
     * @param audioData 5 saniyelik ses verisi (16kHz float array)
     * @return YamnetResult embedding ve baby cry bilgisi ile
     */
    fun process(audioData: FloatArray): YamnetResult {
        val interpreter = this.interpreter ?: throw IllegalStateException("Model not loaded")
        
        // Ses verisini normalize et
        val maxAbs = audioData.maxOfOrNull { kotlin.math.abs(it) } ?: 1f
        val normalizedAudio = if (maxAbs > 0) {
            FloatArray(audioData.size) { audioData[it] / maxAbs }
        } else {
            audioData
        }
        
        // YAMNet her 0.975 saniyelik segmentten embedding çıkarır
        // 5 saniyelik veri = ~5 segment
        val numSegments = normalizedAudio.size / YAMNET_SAMPLE_LENGTH
        
        if (numSegments == 0) {
            return YamnetResult(
                embedding = FloatArray(EMBEDDING_SIZE),
                isBabyCrying = false,
                babyCryScore = 0f,
                topClassName = "Unknown",
                topScore = 0f
            )
        }
        
        // Tüm segmentlerden embedding'leri topla (ortalama için)
        val embeddingSum = FloatArray(EMBEDDING_SIZE)
        val scoresSum = FloatArray(NUM_CLASSES)
        
        for (segmentIdx in 0 until numSegments) {
            val startIdx = segmentIdx * YAMNET_SAMPLE_LENGTH
            val segment = normalizedAudio.sliceArray(startIdx until startIdx + YAMNET_SAMPLE_LENGTH)
            
            // Input tensor hazırla
            val inputBuffer = ByteBuffer.allocateDirect(YAMNET_SAMPLE_LENGTH * 4)
                .order(ByteOrder.nativeOrder())
            for (sample in segment) {
                inputBuffer.putFloat(sample)
            }
            inputBuffer.rewind()
            
            // Output tensorları hazırla
            val scoresOutput = Array(1) { FloatArray(NUM_CLASSES) }
            val embeddingsOutput = Array(1) { FloatArray(EMBEDDING_SIZE) }
            
            // YAMNet çalıştır
            val outputs = mapOf(
                0 to scoresOutput,
                1 to embeddingsOutput
            )
            
            interpreter.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
            
            // Topla
            for (i in 0 until EMBEDDING_SIZE) {
                embeddingSum[i] += embeddingsOutput[0][i]
            }
            for (i in 0 until NUM_CLASSES) {
                scoresSum[i] += scoresOutput[0][i]
            }
        }
        
        // Ortalama al
        val avgEmbedding = FloatArray(EMBEDDING_SIZE) { embeddingSum[it] / numSegments }
        val avgScores = FloatArray(NUM_CLASSES) { scoresSum[it] / numSegments }
        
        // Baby cry kontrolü
        var maxBabyScore = 0f
        for (idx in BABY_CRY_INDICES) {
            if (avgScores[idx] > maxBabyScore) {
                maxBabyScore = avgScores[idx]
            }
        }
        
        // En yüksek skorlu sınıf
        var topIdx = 0
        var topScore = avgScores[0]
        for (i in 1 until NUM_CLASSES) {
            if (avgScores[i] > topScore) {
                topScore = avgScores[i]
                topIdx = i
            }
        }
        
        return YamnetResult(
            embedding = avgEmbedding,
            isBabyCrying = maxBabyScore >= BABY_CRY_THRESHOLD,
            babyCryScore = maxBabyScore,
            topClassName = getClassName(topIdx),
            topScore = topScore
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
     * AudioSet sınıf isimlerinden bazıları
     */
    private fun getClassName(index: Int): String {
        return when (index) {
            0 -> "Speech"
            1 -> "Child speech"
            20 -> "Baby cry"
            21 -> "Infant cry"
            22 -> "Crying"
            137 -> "Music"
            494 -> "Silence"
            else -> "Sound #$index"
        }
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
 * YAMNet işleme sonucu
 */
data class YamnetResult(
    val embedding: FloatArray,      // 1024 boyutlu özellik vektörü
    val isBabyCrying: Boolean,      // Baby cry tespit edildi mi?
    val babyCryScore: Float,        // Baby cry skoru (0-1)
    val topClassName: String,       // En yüksek skorlu sınıf
    val topScore: Float             // En yüksek skor
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as YamnetResult
        return embedding.contentEquals(other.embedding) &&
               isBabyCrying == other.isBabyCrying &&
               babyCryScore == other.babyCryScore
    }

    override fun hashCode(): Int {
        var result = embedding.contentHashCode()
        result = 31 * result + isBabyCrying.hashCode()
        result = 31 * result + babyCryScore.hashCode()
        return result
    }
}
