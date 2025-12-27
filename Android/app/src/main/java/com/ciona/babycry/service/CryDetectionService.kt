package com.ciona.babycry.service

import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.os.Binder
import android.os.Build
import android.os.IBinder
import android.util.Log
import com.ciona.babycry.audio.AudioCapture
import com.ciona.babycry.audio.AudioEvent
import com.ciona.babycry.ml.ClassificationResult
import com.ciona.babycry.ml.CryClassifier
import com.ciona.babycry.ml.YamnetProcessor
import com.ciona.babycry.serial.ArduinoSerial
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.collectLatest

/**
 * CryDetectionService - Arka planda bebek ağlaması tespit servisi
 * 
 * Foreground Service olarak çalışır ve:
 * - Mikrofonu sürekli dinler
 * - YAMNet ile ses analizi yapar
 * - Bebek ağlaması tespit ederse bildirim gönderir
 */
class CryDetectionService : Service() {
    
    companion object {
        private const val TAG = "CryDetectionService"
        const val ACTION_STOP = "com.ciona.babycry.STOP_SERVICE"
        const val ACTION_START = "com.ciona.babycry.START_SERVICE"
        
        private const val DETECTION_WINDOW_MS = 5000L
    }
    
    // Service Binder
    private val binder = LocalBinder()
    
    inner class LocalBinder : Binder() {
        fun getService(): CryDetectionService = this@CryDetectionService
    }
    
    // Modules
    private var audioCapture: AudioCapture? = null
    private var yamnetProcessor: YamnetProcessor? = null
    private var cryClassifier: CryClassifier? = null
    private var arduinoSerial: ArduinoSerial? = null
    
    // Coroutine Scope
    private val serviceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private var processingJob: Job? = null
    
    // State
    private val _isListening = MutableStateFlow(false)
    val isListening: StateFlow<Boolean> = _isListening
    
    private var isProcessing = false
    
    // Detection state
    private val detectionResults = mutableListOf<ClassificationResult>()
    private var detectionStartTime: Long = 0
    private var isCollectingDetections = false
    
    // Notification Manager
    private lateinit var notificationManager: NotificationManager
    
    override fun onCreate() {
        super.onCreate()
        Log.d(TAG, "Service created")
        
        notificationManager = getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
        NotificationHelper.createChannels(this)
        
        initializeModules()
    }
    
    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.d(TAG, "onStartCommand: ${intent?.action}")
        
        when (intent?.action) {
            ACTION_STOP -> {
                Log.d(TAG, "Stop action received")
                stopSelf()
                return START_NOT_STICKY
            }
            else -> {
                // Start foreground service
                startForegroundService()
                startListening()
            }
        }
        
        return START_STICKY
    }
    
    override fun onBind(intent: Intent?): IBinder {
        return binder
    }
    
    override fun onDestroy() {
        Log.d(TAG, "Service destroyed")
        stopListening()
        releaseModules()
        serviceScope.cancel()
        super.onDestroy()
    }
    
    private fun startForegroundService() {
        val notification = NotificationHelper.createServiceNotification(this, true)
        
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            startForeground(
                NotificationHelper.NOTIFICATION_ID_SERVICE,
                notification,
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_MICROPHONE
            )
        } else {
            startForeground(NotificationHelper.NOTIFICATION_ID_SERVICE, notification)
        }
    }
    
    private fun initializeModules() {
        serviceScope.launch(Dispatchers.IO) {
            try {
                Log.d(TAG, "Initializing modules...")
                yamnetProcessor = YamnetProcessor(this@CryDetectionService)
                cryClassifier = CryClassifier(this@CryDetectionService)
                audioCapture = AudioCapture()
                arduinoSerial = ArduinoSerial(this@CryDetectionService)
                
                // Try to connect Arduino
                try {
                    arduinoSerial?.scanAndConnect()
                } catch (e: Exception) {
                    Log.w(TAG, "Arduino connection failed: ${e.message}")
                }
                
                Log.d(TAG, "Modules initialized")
            } catch (e: Exception) {
                Log.e(TAG, "Module initialization failed", e)
            }
        }
    }
    
    private fun startListening() {
        if (_isListening.value) return
        
        serviceScope.launch {
            // Wait for modules to initialize
            var tries = 0
            while (audioCapture == null && tries < 10) {
                delay(500)
                tries++
            }
            
            val capture = audioCapture ?: run {
                Log.e(TAG, "AudioCapture not initialized")
                return@launch
            }
            
            if (!capture.start()) {
                Log.e(TAG, "Failed to start audio capture")
                return@launch
            }
            
            _isListening.value = true
            Log.d(TAG, "Started listening")
            
            // Update notification
            updateServiceNotification(true)
            
            // Send info to Arduino
            withContext(Dispatchers.IO) {
                arduinoSerial?.sendInfo("Dinleniyor...", "Arka Plan Modu")
                arduinoSerial?.setTrafficLight(ArduinoSerial.TrafficLightState.GREEN)
            }
            
            // Start processing audio events
            processingJob = serviceScope.launch {
                capture.audioEvents.collectLatest { event ->
                    when (event) {
                        is AudioEvent.AudioReady -> {
                            if (!isProcessing) {
                                processAudio(event.samples, event.rms)
                            }
                        }
                        is AudioEvent.Silence -> {
                            // Quiet environment
                        }
                        is AudioEvent.Error -> {
                            Log.e(TAG, "Audio error: ${event.message}")
                        }
                    }
                }
            }
        }
    }
    
    private fun stopListening() {
        _isListening.value = false
        processingJob?.cancel()
        processingJob = null
        audioCapture?.stop()
        Log.d(TAG, "Stopped listening")
    }
    
    private fun updateServiceNotification(isListening: Boolean) {
        val notification = NotificationHelper.createServiceNotification(this, isListening)
        notificationManager.notify(NotificationHelper.NOTIFICATION_ID_SERVICE, notification)
    }
    
    private suspend fun processAudio(samples: FloatArray, rms: Float) {
        val yamnet = yamnetProcessor ?: return
        val classifier = cryClassifier ?: return
        
        isProcessing = true
        
        try {
            val yamnetResult = withContext(Dispatchers.Default) {
                yamnet.process(samples)
            }
            
            // Not baby crying - skip
            if (!yamnetResult.isBabyCrying) {
                if (isCollectingDetections) {
                    val elapsed = System.currentTimeMillis() - detectionStartTime
                    if (elapsed > 2000 && detectionResults.isEmpty()) {
                        isCollectingDetections = false
                    }
                }
                
                // Update Arduino
                withContext(Dispatchers.IO) {
                    arduinoSerial?.setTrafficLight(ArduinoSerial.TrafficLightState.GREEN)
                }
                
                isProcessing = false
                return
            }
            
            // Baby cry detected!
            Log.d(TAG, "Baby cry detected!")
            
            withContext(Dispatchers.IO) {
                arduinoSerial?.setTrafficLight(ArduinoSerial.TrafficLightState.RED)
            }
            
            val result = withContext(Dispatchers.Default) {
                classifier.classify(yamnetResult.embedding)
            }
            
            if (!isCollectingDetections) {
                isCollectingDetections = true
                detectionStartTime = System.currentTimeMillis()
                detectionResults.clear()
            }
            
            detectionResults.add(result)
            
            // Check if detection window complete
            val elapsed = System.currentTimeMillis() - detectionStartTime
            if (elapsed >= DETECTION_WINDOW_MS && detectionResults.isNotEmpty()) {
                val finalResult = computeFinalResult(detectionResults)
                val secondBest = findSecondBest(detectionResults)
                
                Log.d(TAG, "Final result: ${finalResult.predictedLabel} (${finalResult.confidence})")
                
                // Send detection notification
                sendDetectionNotification(finalResult, secondBest)
                
                // Send to Arduino
                withContext(Dispatchers.IO) {
                    val conf = (finalResult.confidence * 100).toInt()
                    arduinoSerial?.sendInfo(finalResult.predictedLabel, "$conf Guven")
                    
                    // Play lullaby for tired/discomfort
                    if (finalResult.predictedClass in listOf("tired", "discomfort")) {
                        arduinoSerial?.sendInfo("Ninni Caliyor", "Dandini Dastana")
                        arduinoSerial?.playLullaby()
                    }
                }
                
                // Reset detection state
                isCollectingDetections = false
                detectionResults.clear()
                
                // Small delay before continuing
                delay(5000)
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Process audio error", e)
        } finally {
            isProcessing = false
        }
    }
    
    private fun sendDetectionNotification(result: ClassificationResult, secondBest: String?) {
        val notification = NotificationHelper.createDetectionNotification(this, result, secondBest)
        notificationManager.notify(NotificationHelper.NOTIFICATION_ID_DETECTION, notification)
    }
    
    /**
     * Find second best prediction from results
     */
    private fun findSecondBest(results: List<ClassificationResult>): String? {
        val votes = mutableMapOf<String, Float>()
        for (r in results) {
            votes[r.predictedClass] = (votes[r.predictedClass] ?: 0f) + r.confidence
        }
        val sorted = votes.entries.sortedByDescending { it.value }
        return if (sorted.size > 1) {
            val label = sorted[1].key
            CryClassifier.CLASS_LABELS_TR[label] ?: label
        } else null
    }
    
    private fun computeFinalResult(results: List<ClassificationResult>): ClassificationResult {
        if (results.isEmpty()) {
            return ClassificationResult(
                predictedClass = "unknown",
                predictedLabel = "Unknown",
                emoji = "❓",
                confidence = 0f,
                isConfident = false,
                allProbabilities = emptyMap()
            )
        }
        
        val votes = mutableMapOf<String, Float>()
        
        for (result in results) {
            val cls = result.predictedClass
            votes[cls] = (votes[cls] ?: 0f) + result.confidence
        }
        
        val winningClass = votes.maxByOrNull { it.value }?.key ?: results.last().predictedClass
        val classResults = results.filter { it.predictedClass == winningClass }
        val avgConfidence = classResults.map { it.confidence }.average().toFloat()
        
        val avgProbs = mutableMapOf<String, Float>()
        val sampleCount = classResults.size
        if (sampleCount > 0) {
            for (result in classResults) {
                for ((k, v) in result.allProbabilities) {
                    avgProbs[k] = (avgProbs[k] ?: 0f) + v / sampleCount
                }
            }
        }
        
        val bestResult = classResults.maxByOrNull { it.confidence } ?: results.last()
        
        return ClassificationResult(
            predictedClass = winningClass,
            predictedLabel = bestResult.predictedLabel,
            emoji = bestResult.emoji,
            confidence = avgConfidence,
            isConfident = avgConfidence >= CryClassifier.CONFIDENCE_THRESHOLD,
            allProbabilities = avgProbs
        )
    }
    
    private fun releaseModules() {
        stopListening()
        yamnetProcessor?.close()
        cryClassifier?.close()
        arduinoSerial?.release()
        
        yamnetProcessor = null
        cryClassifier = null
        audioCapture = null
        arduinoSerial = null
    }
}
