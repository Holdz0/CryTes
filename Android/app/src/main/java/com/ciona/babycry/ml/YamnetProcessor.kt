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
 * YamnetProcessor - YAMNet TFLite ile ses embedding'i çıkarma
 * 
 * YAMNet TFLite Outputs:
 * - Output 0: [1, 521] - AudioSet class scores
 * - Output 1: [1, 1024] - Embeddings (for classifier)
 * - Output 2: [1, 64] - Spectrogram features (ignored)
 */
class YamnetProcessor(context: Context) {
    
    companion object {
        private const val TAG = "YamnetProcessor"
        private const val MODEL_FILE = "yamnet.tflite"
        private const val YAMNET_SAMPLE_LENGTH = 15600  // ~0.975 saniye @ 16kHz
        private const val EMBEDDING_SIZE = 1024
        private const val NUM_CLASSES = 521
        
        // Baby cry sınıf indeksleri (YAMNet AudioSet)
        // 20 = Baby cry, infant cry
        // 21 = Crying, sobbing  
        // 22 = Whimper
        val BABY_CRY_INDICES = intArrayOf(20, 21, 22)
        const val BABY_CRY_THRESHOLD = 0.05f  // %5 eşik
    }
    
    private var interpreter: Interpreter? = null
    
    init {
        try {
            val model = loadModelFile(context, MODEL_FILE)
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            interpreter = Interpreter(model, options)
            logModelInfo()
            Log.d(TAG, "YAMNet model loaded successfully")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load YAMNet model", e)
            throw e
        }
    }
    
    private fun logModelInfo() {
        val interp = interpreter ?: return
        
        // Input info
        val inputTensor = interp.getInputTensor(0)
        Log.d(TAG, "Input: shape=${inputTensor.shape().contentToString()}, type=${inputTensor.dataType()}")
        
        // Output info
        val outputCount = interp.outputTensorCount
        Log.d(TAG, "YAMNet has $outputCount outputs")
        
        for (i in 0 until outputCount) {
            val tensor = interp.getOutputTensor(i)
            Log.d(TAG, "Output $i: shape=${tensor.shape().contentToString()}, type=${tensor.dataType()}")
        }
    }
    
    fun process(audioData: FloatArray): YamnetResult {
        val interp = interpreter ?: throw IllegalStateException("Model not loaded")
        
        try {
            Log.d(TAG, "Processing audio: ${audioData.size} samples")
            
            // 1. Normalize audio to [-1, 1]
            val maxAbs = audioData.maxOfOrNull { kotlin.math.abs(it) } ?: 1f
            val normalizedAudio = if (maxAbs > 0.001f) {
                FloatArray(audioData.size) { audioData[it] / maxAbs }
            } else {
                Log.w(TAG, "Audio is nearly silent (maxAbs=$maxAbs)")
                audioData
            }
            
            // 2. Check if we have enough data
            val numSegments = normalizedAudio.size / YAMNET_SAMPLE_LENGTH
            if (numSegments == 0) {
                Log.w(TAG, "Not enough audio data: ${normalizedAudio.size} < $YAMNET_SAMPLE_LENGTH")
                return createEmptyResult()
            }
            
            // 3. Use the last segment (most recent audio)
            val lastSegmentStart = (numSegments - 1) * YAMNET_SAMPLE_LENGTH
            val segment = normalizedAudio.sliceArray(lastSegmentStart until lastSegmentStart + YAMNET_SAMPLE_LENGTH)
            Log.d(TAG, "Using segment from $lastSegmentStart, length=${segment.size}")
            
            // 4. CRITICAL: Resize input tensor to match our segment size
            interp.resizeInput(0, intArrayOf(segment.size))
            interp.allocateTensors()
            
            // 5. Prepare input buffer
            val inputBuffer = ByteBuffer.allocateDirect(segment.size * 4)
                .order(ByteOrder.nativeOrder())
            for (sample in segment) {
                inputBuffer.putFloat(sample)
            }
            inputBuffer.rewind()
            
            // 6. Prepare output buffers - FIXED STRUCTURE based on model analysis
            // Output 0: [1, 521] scores
            val scoresOutput = Array(1) { FloatArray(NUM_CLASSES) }
            // Output 1: [1, 1024] embeddings  
            val embeddingsOutput = Array(1) { FloatArray(EMBEDDING_SIZE) }
            // Output 2: [1, 64] spectrogram (we ignore this)
            val spectrogramOutput = Array(1) { FloatArray(64) }
            
            val outputs = mapOf(
                0 to scoresOutput,
                1 to embeddingsOutput,
                2 to spectrogramOutput
            )
            
            // 7. Run inference
            Log.d(TAG, "Running YAMNet inference...")
            interp.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputs)
            
            // 8. Extract results directly
            val scores = scoresOutput[0]  // FloatArray(521)
            val embeddings = embeddingsOutput[0]  // FloatArray(1024)
            
            // Debug: Check if outputs are valid
            val maxScore = scores.maxOrNull() ?: 0f
            val embeddingSum = embeddings.sum()
            Log.d(TAG, "Scores max=$maxScore, Embedding sum=$embeddingSum")
            
            if (maxScore == 0f && embeddingSum == 0f) {
                Log.w(TAG, "WARNING: All outputs are zero! Model may not be working correctly.")
            }
            
            return createResult(scores, embeddings)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error processing audio", e)
            e.printStackTrace()
            return createEmptyResult()
        }
    }
    
    private fun createResult(scores: FloatArray, embedding: FloatArray): YamnetResult {
        // Baby cry check - look at specific AudioSet indices
        var maxBabyScore = 0f
        for (idx in BABY_CRY_INDICES) {
            if (idx < scores.size) {
                val score = scores[idx]
                Log.d(TAG, "Baby index $idx score: ${score * 100}%")
                if (score > maxBabyScore) {
                    maxBabyScore = score
                }
            }
        }
        
        // Find top class
        var topIdx = 0
        var topScore = if (scores.isNotEmpty()) scores[0] else 0f
        for (i in 1 until scores.size) {
            if (scores[i] > topScore) {
                topScore = scores[i]
                topIdx = i
            }
        }
        
        val className = getClassName(topIdx)
        Log.d(TAG, "🎯 Top: $className ($topIdx) = ${(topScore * 100).toInt()}%, Baby Score: ${(maxBabyScore * 100).toInt()}%")
        
        val isBabyCrying = maxBabyScore >= BABY_CRY_THRESHOLD
        if (isBabyCrying) {
            Log.d(TAG, "👶 BABY CRY DETECTED! Score: ${(maxBabyScore * 100).toInt()}%")
        }
        
        return YamnetResult(
            embedding = embedding,
            isBabyCrying = isBabyCrying,
            babyCryScore = maxBabyScore,
            topClassName = className,
            topScore = topScore
        )
    }
    
    private fun createEmptyResult(): YamnetResult {
        return YamnetResult(
            embedding = FloatArray(EMBEDDING_SIZE),
            isBabyCrying = false,
            babyCryScore = 0f,
            topClassName = "Silence",
            topScore = 0f
        )
    }
    
    private fun loadModelFile(context: Context, filename: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(filename)
        val inputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
    }
    
    private fun getClassName(index: Int): String {
        return AUDIOSET_CLASSES.getOrElse(index) { "Sound #$index" }
    }
    
    fun close() {
        interpreter?.close()
        interpreter = null
    }
    
    // AudioSet class names (commonly detected ones)
    private val AUDIOSET_CLASSES = mapOf(
        0 to "Speech",
        1 to "Child speech",
        2 to "Conversation",
        3 to "Narration",
        4 to "Babbling",
        5 to "Speech synthesizer",
        6 to "Shout",
        7 to "Bellow",
        8 to "Whoop",
        9 to "Yell",
        10 to "Children shouting",
        11 to "Screaming",
        12 to "Whispering",
        13 to "Laughter",
        14 to "Baby laughter",
        15 to "Giggle",
        16 to "Snicker",
        17 to "Belly laugh",
        18 to "Chuckle",
        19 to "Crying",
        20 to "Baby cry",
        21 to "Whimper",
        22 to "Wail",
        23 to "Sigh",
        24 to "Singing",
        25 to "Choir",
        26 to "Yodeling",
        27 to "Chant",
        28 to "Mantra",
        29 to "Child singing",
        30 to "Synthetic singing",
        31 to "Rapping",
        32 to "Humming",
        33 to "Groan",
        34 to "Grunt",
        35 to "Whistling",
        36 to "Breathing",
        37 to "Wheeze",
        38 to "Snoring",
        39 to "Gasp",
        40 to "Pant",
        41 to "Snort",
        42 to "Cough",
        43 to "Throat clearing",
        44 to "Sneeze",
        45 to "Sniff",
        46 to "Run",
        47 to "Shuffle",
        48 to "Walk",
        49 to "Footsteps",
        50 to "Chewing",
        137 to "Music",
        494 to "Silence",
        500 to "Inside",
        501 to "Outside"
    )
}

data class YamnetResult(
    val embedding: FloatArray,
    val isBabyCrying: Boolean,
    val babyCryScore: Float,
    val topClassName: String,
    val topScore: Float
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
