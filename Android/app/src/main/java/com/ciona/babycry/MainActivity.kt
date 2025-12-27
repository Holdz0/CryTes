package com.ciona.babycry

import android.Manifest
import android.animation.ObjectAnimator
import android.animation.PropertyValuesHolder
import android.content.pm.PackageManager
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.animation.AccelerateDecelerateInterpolator
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.GravityCompat
import androidx.drawerlayout.widget.DrawerLayout
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.ciona.babycry.audio.AudioCapture
import com.ciona.babycry.audio.AudioEvent
import com.ciona.babycry.databinding.ActivityMainBinding
import com.ciona.babycry.ml.ClassificationResult
import com.ciona.babycry.ml.CryClassifier
import com.ciona.babycry.ml.YamnetProcessor
import com.ciona.babycry.model.CryHistory
import com.ciona.babycry.serial.ArduinoSerial
import com.ciona.babycry.ui.CryHistoryAdapter
import android.hardware.usb.UsbManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.widget.Toast
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.collectLatest

/**
 * MainActivity - Baby Cry Analyzer
 */
class MainActivity : AppCompatActivity() {
    
    companion object {
        private const val TAG = "MainActivity"
        private const val DETECTION_WINDOW_MS = 5000L
        private const val SERIAL_SEND_INTERVAL = 1000L
    }
    
    private lateinit var binding: ActivityMainBinding
    
    // Animation
    private var pulseAnimator: ObjectAnimator? = null
    private var ringAnimator1: ObjectAnimator? = null
    private var ringAnimator2: ObjectAnimator? = null
    private var ringAnimator3: ObjectAnimator? = null
    
    // Adapters
    private lateinit var historyAdapter: CryHistoryAdapter
    private lateinit var drawerHistoryAdapter: CryHistoryAdapter
    
    // Modules
    private var audioCapture: AudioCapture? = null
    private var yamnetProcessor: YamnetProcessor? = null
    private var cryClassifier: CryClassifier? = null
    private var arduinoSerial: ArduinoSerial? = null
    
    // State
    private var isListening = false
    private var isPaused = false
    private var processingJob: Job? = null
    private var isProcessing = false
    
    // Detection
    private val detectionResults = mutableListOf<ClassificationResult>()
    private var detectionStartTime: Long = 0
    private var isCollectingDetections = false
    private var lastSerialSendTime = 0L
    
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startListening()
        } else {
            updateStatus("Microphone permission required", "Please grant permission to use the app.")
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        setupUI()
        setupDrawer()
        setupHistoryRecyclerViews()
        setupPulseAnimation()
        setupSensitivitySlider()
        
        initializeModules()
        registerUsbReceiver()
        
        // Arduino reconnect
        binding.arduinoStatus.setOnClickListener {
            reconnectArduino()
        }
        
        // Settings button - opens activities dialog
        binding.settingsButton.setOnClickListener {
            com.ciona.babycry.ui.ActivitiesDialog(
                this,
                arduinoSerial,
                lifecycleScope
            ).show()
        }
        
        // Action button (Pause/Resume)
        binding.actionButton.setOnClickListener {
            togglePause()
        }
        
        // View All button opens drawer
        binding.viewAllButton.setOnClickListener {
            binding.drawerLayout.openDrawer(GravityCompat.START)
        }
        
        // Bottom nav - History opens drawer
        binding.navHistory.setOnClickListener {
            binding.drawerLayout.openDrawer(GravityCompat.START)
        }
    }
    
    private fun setupUI() {
        // Set initial sensitivity progress (75%)
        binding.sensitivityProgress.post {
            val params = binding.sensitivityProgress.layoutParams
            params.width = (binding.sensitivityProgress.parent as View).width * 75 / 100
            binding.sensitivityProgress.layoutParams = params
        }
    }
    
    private fun setupDrawer() {
        binding.menuButton.setOnClickListener {
            if (binding.drawerLayout.isDrawerOpen(GravityCompat.START)) {
                binding.drawerLayout.closeDrawer(GravityCompat.START)
            } else {
                binding.drawerLayout.openDrawer(GravityCompat.START)
            }
        }
    }
    
    private fun setupHistoryRecyclerViews() {
        historyAdapter = CryHistoryAdapter()
        binding.historyRecyclerView.apply {
            layoutManager = LinearLayoutManager(this@MainActivity)
            adapter = historyAdapter
        }
        
        drawerHistoryAdapter = CryHistoryAdapter()
        binding.drawerHistoryRecyclerView.apply {
            layoutManager = LinearLayoutManager(this@MainActivity)
            adapter = drawerHistoryAdapter
        }
        
        updateHistoryVisibility()
    }
    
    private fun updateHistoryVisibility() {
        if (historyAdapter.isEmpty()) {
            binding.historyRecyclerView.visibility = View.GONE
            binding.emptyHistoryContainer.visibility = View.VISIBLE
        } else {
            binding.historyRecyclerView.visibility = View.VISIBLE
            binding.emptyHistoryContainer.visibility = View.GONE
        }
    }
    
    private fun addToHistory(result: ClassificationResult) {
        val history = CryHistory(
            cryType = result.predictedClass,
            cryLabel = result.predictedLabel,
            emoji = result.emoji,
            confidence = result.confidence
        )
        historyAdapter.addHistory(history)
        drawerHistoryAdapter.addHistory(history)
        updateHistoryVisibility()
    }
    
    private fun setupPulseAnimation() {
        // Pulse animation for rings
        val scaleX = PropertyValuesHolder.ofFloat(View.SCALE_X, 0.95f, 1.05f, 0.95f)
        val scaleY = PropertyValuesHolder.ofFloat(View.SCALE_Y, 0.95f, 1.05f, 0.95f)
        val alpha = PropertyValuesHolder.ofFloat(View.ALPHA, 0.7f, 1f, 0.7f)
        
        ringAnimator1 = ObjectAnimator.ofPropertyValuesHolder(binding.pulseRingOuter, scaleX, scaleY, alpha).apply {
            duration = 2000
            repeatCount = ObjectAnimator.INFINITE
            interpolator = AccelerateDecelerateInterpolator()
        }
        
        ringAnimator2 = ObjectAnimator.ofPropertyValuesHolder(binding.pulseRingMiddle, scaleX, scaleY, alpha).apply {
            duration = 1800
            repeatCount = ObjectAnimator.INFINITE
            interpolator = AccelerateDecelerateInterpolator()
            startDelay = 200
        }
        
        ringAnimator3 = ObjectAnimator.ofPropertyValuesHolder(binding.pulseRingInner, scaleX, scaleY, alpha).apply {
            duration = 1600
            repeatCount = ObjectAnimator.INFINITE
            interpolator = AccelerateDecelerateInterpolator()
            startDelay = 400
        }
    }
    
    private fun startPulseAnimation() {
        ringAnimator1?.start()
        ringAnimator2?.start()
        ringAnimator3?.start()
    }
    
    private fun stopPulseAnimation() {
        ringAnimator1?.cancel()
        ringAnimator2?.cancel()
        ringAnimator3?.cancel()
        
        binding.pulseRingOuter.scaleX = 1f
        binding.pulseRingOuter.scaleY = 1f
        binding.pulseRingMiddle.scaleX = 1f
        binding.pulseRingMiddle.scaleY = 1f
        binding.pulseRingInner.scaleX = 1f
        binding.pulseRingInner.scaleY = 1f
    }
    
    private fun setupSensitivitySlider() {
        // Static for now, can be made interactive later
    }
    
    private fun togglePause() {
        if (isPaused) {
            // Resume
            isPaused = false
            binding.actionButtonText.text = "Pause Monitoring"
            startListening()
        } else {
            // Pause
            isPaused = true
            binding.actionButtonText.text = "Resume Monitoring"
            stopListening()
            updateStatus("Paused", "Monitoring is paused.\nTap Resume to continue.")
        }
    }
    
    private fun reconnectArduino() {
        updateStatus("Connecting...", "Searching for Arduino...")
        lifecycleScope.launch(Dispatchers.IO) {
            arduinoSerial?.disconnect()
            delay(500)
            val connected = arduinoSerial?.scanAndConnect() == true
            withContext(Dispatchers.Main) {
                if (connected) {
                    Toast.makeText(this@MainActivity, "Arduino Connected", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this@MainActivity, "Arduino Not Found", Toast.LENGTH_SHORT).show()
                }
                if (!isPaused) {
                    updateStatus("Listening...", "Monitoring active in background.\nEnvironment is quiet.")
                }
            }
        }
    }
    
    override fun onResume() {
        super.onResume()
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                arduinoSerial?.scanAndConnect()
            } catch (e: Exception) {
                Log.e(TAG, "Arduino scan error", e)
            }
        }
    }
    
    override fun onPause() {
        super.onPause()
        stopListening()
        stopPulseAnimation()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        ringAnimator1?.cancel()
        ringAnimator2?.cancel()
        ringAnimator3?.cancel()
        unregisterUsbReceiver()
        releaseModules()
    }
    
    // USB Receiver
    private val usbReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                UsbManager.ACTION_USB_DEVICE_ATTACHED -> {
                    Log.d(TAG, "USB Device Attached")
                    Toast.makeText(context, "USB Device Detected", Toast.LENGTH_SHORT).show()
                    stopListening()
                    lifecycleScope.launch(Dispatchers.IO) {
                        delay(1000)
                        arduinoSerial?.scanAndConnect()
                        withContext(Dispatchers.Main) {
                            if (!isPaused) startListening()
                        }
                    }
                }
                UsbManager.ACTION_USB_DEVICE_DETACHED -> {
                    Log.d(TAG, "USB Device Detached")
                    Toast.makeText(context, "USB Device Removed", Toast.LENGTH_SHORT).show()
                    arduinoSerial?.disconnect()
                }
            }
        }
    }
    
    private fun registerUsbReceiver() {
        val filter = IntentFilter().apply {
            addAction(UsbManager.ACTION_USB_DEVICE_ATTACHED)
            addAction(UsbManager.ACTION_USB_DEVICE_DETACHED)
        }
        if (android.os.Build.VERSION.SDK_INT >= android.os.Build.VERSION_CODES.TIRAMISU) {
            registerReceiver(usbReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            registerReceiver(usbReceiver, filter)
        }
    }
    
    private fun unregisterUsbReceiver() {
        try {
            unregisterReceiver(usbReceiver)
        } catch (e: Exception) {
            Log.e(TAG, "Receiver unregister error", e)
        }
    }
    
    private fun initializeModules() {
        updateStatus("Loading...", "Initializing models...")
        
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                Log.d(TAG, "Loading YAMNet...")
                yamnetProcessor = YamnetProcessor(this@MainActivity)
                
                Log.d(TAG, "Loading CryClassifier...")
                cryClassifier = CryClassifier(this@MainActivity)
                
                Log.d(TAG, "Initializing AudioCapture...")
                audioCapture = AudioCapture()
                
                Log.d(TAG, "Initializing ArduinoSerial...")
                arduinoSerial = ArduinoSerial(this@MainActivity)
                
                withContext(Dispatchers.Main) {
                    observeArduinoConnection()
                    updateStatus("Ready", "Models loaded successfully.")
                    checkPermissionAndStart()
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Module initialization error", e)
                withContext(Dispatchers.Main) {
                    updateStatus("Error", "Failed to load models: ${e.message?.take(50)}")
                }
            }
        }
    }
    
    private fun checkPermissionAndStart() {
        when {
            ContextCompat.checkSelfPermission(
                this, Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED -> {
                startListening()
            }
            else -> {
                permissionLauncher.launch(Manifest.permission.RECORD_AUDIO)
            }
        }
    }
    
    private fun startListening() {
        if (isListening || isPaused) return
        
        val capture = audioCapture ?: return
        
        if (!capture.start()) {
            updateStatus("Error", "Could not start microphone.")
            return
        }
        
        isListening = true
        updateStatus("Listening...", "Monitoring active in background.\nEnvironment is quiet.")
        startPulseAnimation()
        
        // Update status badge
        binding.statusBadgeText.text = "LIVE FEED"
        binding.statusDot.setBackgroundResource(R.drawable.circle_small)
        
        lifecycleScope.launch(Dispatchers.IO) {
            arduinoSerial?.sendInfo("Listening...", "Waiting for baby")
            arduinoSerial?.setTrafficLight(ArduinoSerial.TrafficLightState.GREEN)
        }
        
        processingJob = lifecycleScope.launch {
            try {
                capture.audioEvents.collectLatest { event ->
                    when (event) {
                        is AudioEvent.AudioReady -> {
                            if (!isProcessing) {
                                processAudio(event.samples, event.rms)
                            }
                        }
                        is AudioEvent.Silence -> {
                            updateDebug("RMS: ${String.format("%.4f", event.rms)} (quiet)")
                            withContext(Dispatchers.IO) {
                                arduinoSerial?.setTrafficLight(ArduinoSerial.TrafficLightState.GREEN)
                            }
                        }
                        is AudioEvent.Error -> {
                            Log.e(TAG, "Audio Error: ${event.message}")
                            withContext(Dispatchers.Main) {
                                updateDebug("Mic Error: ${event.message}")
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Audio processing error", e)
                withContext(Dispatchers.Main) {
                    updateDebug("Error: ${e.message}")
                }
            }
        }
    }
    
    private fun stopListening() {
        isListening = false
        processingJob?.cancel()
        processingJob = null
        audioCapture?.stop()
        stopPulseAnimation()
        
        binding.statusBadgeText.text = "PAUSED"
    }
    
    private suspend fun processAudio(samples: FloatArray, rms: Float) {
        val yamnet = yamnetProcessor ?: return
        val classifier = cryClassifier ?: return
        
        isProcessing = true
        
        try {
            withContext(Dispatchers.Main) {
                updateDebug("RMS: ${String.format("%.4f", rms)} | Processing...")
            }
            
            val yamnetResult = withContext(Dispatchers.Default) {
                yamnet.process(samples)
            }
            
            withContext(Dispatchers.Main) {
                updateDebug("RMS: ${String.format("%.4f", rms)} | ${yamnetResult.topClassName} (${String.format("%.1f", yamnetResult.topScore * 100)}%)")
            }
            
            if (!yamnetResult.isBabyCrying) {
                if (isCollectingDetections) {
                    val elapsed = System.currentTimeMillis() - detectionStartTime
                    if (elapsed > 2000 && detectionResults.isEmpty()) {
                        isCollectingDetections = false
                        withContext(Dispatchers.Main) {
                            updateStatus("Listening...", "Monitoring active in background.\nEnvironment is quiet.")
                            startPulseAnimation()
                        }
                    }
                }
                
                val now = System.currentTimeMillis()
                if (now - lastSerialSendTime >= SERIAL_SEND_INTERVAL) {
                    lastSerialSendTime = now
                    try {
                        val labelToSend = when(yamnetResult.topClassName) {
                            "Speech" -> "Konusma"
                            "Silence" -> "Sessizlik"
                            "Music" -> "Muzik"
                            else -> yamnetResult.topClassName
                        }
                        val score = (yamnetResult.topScore * 100).toInt()
                        arduinoSerial?.sendInfo("Ses: $labelToSend", "%$score - Bebek yok")
                        arduinoSerial?.setTrafficLight(ArduinoSerial.TrafficLightState.YELLOW)
                    } catch (e: Exception) {
                        Log.e(TAG, "Non-cry serial error", e)
                    }
                }
                
                isProcessing = false
                return
            }
            
            // Baby cry detected!
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
                
                withContext(Dispatchers.Main) {
                    updateStatus("Analyzing...", "Baby cry detected!\nAnalyzing the cause...")
                    stopPulseAnimation()
                    binding.statusBadgeText.text = "DETECTING"
                }
            }
            
            detectionResults.add(result)
            
            val elapsedSeconds = (System.currentTimeMillis() - detectionStartTime) / 1000
            withContext(Dispatchers.Main) {
                updateStatus("Analyzing...", "Processing... (${elapsedSeconds}s / 5s)")
            }
            
            val elapsed = System.currentTimeMillis() - detectionStartTime
            if (elapsed >= DETECTION_WINDOW_MS && detectionResults.isNotEmpty()) {
                stopListening()
                
                lifecycleScope.launch {
                    try {
                        val finalResult = computeFinalResult(detectionResults)
                        addToHistory(finalResult)
                        
                        val secondBest = findSecondBest(detectionResults)
                        
                        Log.d(TAG, "Final decision: ${finalResult.predictedLabel}")
                        
                        withContext(Dispatchers.IO) {
                            try {
                                val conf = (finalResult.confidence * 100).toInt()
                                arduinoSerial?.sendInfo(finalResult.predictedLabel, "$conf Guven")
                            } catch (e: Exception) {
                                Log.e(TAG, "Arduino send error", e)
                            }
                        }
                        
                        withContext(Dispatchers.Main) {
                            updateStatus("${finalResult.emoji} ${finalResult.predictedLabel}", "Detected with ${String.format("%.0f", finalResult.confidence * 100)}% confidence")
                        }
                        
                        val sensorReading = arduinoSerial?.readSensorData()
                        
                        if (sensorReading != null) {
                            val warnings = checkEnvironment(sensorReading.temp, sensorReading.hum)
                            if (warnings.isNotEmpty()) {
                                withContext(Dispatchers.IO) {
                                    for ((title, message) in warnings) {
                                        arduinoSerial?.sendScrollingInfo(title, message, 5000)
                                    }
                                }
                            }
                        }
                        
                        if (finalResult.predictedClass in listOf("tired", "discomfort")) {
                            Log.d(TAG, "Baby tired/uncomfortable - Starting lullaby...")
                            withContext(Dispatchers.IO) {
                                arduinoSerial?.sendInfo("Ninni Caliyor", "Dandini Dastana")
                                arduinoSerial?.playLullaby()
                            }
                            delay(30000)
                            Log.d(TAG, "Lullaby finished")
                        }
                        
                        withContext(Dispatchers.Main) {
                            com.ciona.babycry.ui.FollowUpDialog(
                                this@MainActivity,
                                finalResult,
                                sensorReading?.displayString,
                                secondBest
                            ) {
                                lifecycleScope.launch {
                                    updateStatus("Restarting...", "Preparing to listen again...")
                                    delay(1000)
                                    audioCapture?.clearBuffer()
                                    isCollectingDetections = false
                                    detectionResults.clear()
                                    startListening()
                                }
                            }.show()
                        }
                    } catch (e: Exception) {
                        Log.e(TAG, "Result processing error", e)
                        updateStatus("Error", "Processing failed, restarting...")
                        delay(2000)
                        startListening()
                    }
                }
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Process audio error", e)
            withContext(Dispatchers.Main) {
                updateDebug("Processing error: ${e.message?.take(30)}")
            }
            if (!isListening) startListening()
        } finally {
            isProcessing = false
        }
    }
    
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
    
    private fun checkEnvironment(temp: Float, hum: Float): List<Pair<String, String>> {
        val warnings = mutableListOf<Pair<String, String>>()
        val TEMP_HIGH = 28.0f
        val TEMP_LOW = 18.0f
        val HUM_HIGH = 70.0f
        val HUM_LOW = 30.0f
        
        if (temp > TEMP_HIGH) {
            warnings.add("Terliyor Olabilir" to "Sicak ${temp.toInt()}C")
        } else if (temp < TEMP_LOW) {
            warnings.add("Usuyor Olabilir" to "Soguk ${temp.toInt()}C")
        }
        
        if (hum > HUM_HIGH) {
            warnings.add("Terliyor Olabilir" to "Nem Yuksek %${hum.toInt()}")
        } else if (hum < HUM_LOW) {
            warnings.add("Kuru Hava Uyarisi" to "Nem Dusuk %${hum.toInt()}")
        }
        
        return warnings
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
        val probSums = mutableMapOf<String, MutableMap<String, Float>>()
        
        for (result in results) {
            val cls = result.predictedClass
            votes[cls] = (votes[cls] ?: 0f) + result.confidence
            
            if (!probSums.containsKey(cls)) {
                probSums[cls] = mutableMapOf()
            }
            for ((k, v) in result.allProbabilities) {
                probSums[cls]!![k] = (probSums[cls]!![k] ?: 0f) + v
            }
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
    
    private fun updateStatus(title: String, subtitle: String) {
        binding.statusText.text = title
        binding.statusSubtext.text = subtitle
    }
    
    private fun updateDebug(text: String) {
        binding.debugText.text = text
    }
    
    private fun observeArduinoConnection() {
        lifecycleScope.launch {
            try {
                arduinoSerial?.isConnected?.collect { connected ->
                    val indicator = binding.arduinoIndicator.background as? GradientDrawable
                    if (connected) {
                        indicator?.setColor(getColor(R.color.success))
                        binding.arduinoText.text = getString(R.string.arduino_connected)
                    } else {
                        indicator?.setColor(getColor(R.color.error))
                        binding.arduinoText.text = getString(R.string.arduino_disconnected)
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Arduino observe error", e)
            }
        }
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
