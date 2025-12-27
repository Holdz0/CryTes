package com.ciona.babycry

import android.Manifest
import android.animation.ObjectAnimator
import android.animation.PropertyValuesHolder
import android.content.pm.PackageManager
import android.graphics.drawable.GradientDrawable
import android.os.Build
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
import com.ciona.babycry.model.HistoryManager
import com.ciona.babycry.serial.ArduinoSerial
import com.ciona.babycry.service.CryDetectionService
import com.ciona.babycry.service.NotificationHelper
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
        private const val PREFS_NAME = "crytes_prefs"
        private const val PREF_LISTENING_ENABLED = "listening_enabled"
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
            // MainActivity açıkken kendi tespitini yapacak, service başlatmıyoruz
            startListening()
        } else {
            updateStatus("Microphone permission required", "Please grant permission to use the app.")
        }
    }
    
    private val notificationPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        // Bildirim izni verilse de verilmese de devam et
        // Service başlatmıyoruz - sadece arka plandayken çalışacak
        checkPermissionAndStart()
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // Create notification channels
        NotificationHelper.createChannels(this)
        
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
        
        // Bottom nav - Guides opens guides dialog
        binding.navGuides.setOnClickListener {
            com.ciona.babycry.ui.GuidesDialog(this).show()
        }
        
        // Handle notification click intent
        handleNotificationIntent(intent)
    }
    
    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        handleNotificationIntent(intent)
    }
    
    /**
     * Handle notification deep link - show FollowUpDialog directly
     */
    private fun handleNotificationIntent(intent: Intent?) {
        val cryType = intent?.getStringExtra("CRY_TYPE") ?: return
        val cryLabel = intent.getStringExtra("CRY_LABEL") ?: return
        val cryEmoji = intent.getStringExtra("CRY_EMOJI") ?: "❓"
        val cryConfidence = intent.getFloatExtra("CRY_CONFIDENCE", 0.5f)
        val secondBest = intent.getStringExtra("CRY_SECOND_BEST")
        
        Log.d(TAG, "Notification click: $cryType, secondBest: $secondBest")
        
        // Create result from intent
        val result = ClassificationResult(
            predictedClass = cryType,
            predictedLabel = cryLabel,
            emoji = cryEmoji,
            confidence = cryConfidence,
            isConfident = cryConfidence >= CryClassifier.CONFIDENCE_THRESHOLD,
            allProbabilities = emptyMap()
        )
        
        // Add to history
        addToHistory(result)
        
        // Show FollowUpDialog directly with secondBest
        lifecycleScope.launch {
            delay(500) // Small delay to ensure activity is ready
            val displayConfidence = NotificationHelper.adjustConfidenceForDisplay(result.confidence)
            updateStatus("${result.emoji} ${result.predictedLabel}", "%$displayConfidence güvenle tespit edildi")
            
            com.ciona.babycry.ui.FollowUpDialog(
                this@MainActivity,
                result,
                null,
                secondBest
            ) {
                // On complete - continue listening
                lifecycleScope.launch {
                    updateStatus("Dinleniyor...", "Arka planda dinleme aktif.\nOrtam sessiz.")
                }
            }.show()
        }
        
        // Clear the intent extras to avoid re-processing
        intent?.removeExtra("CRY_TYPE")
    }
    
    /**
     * Start background detection service
     */
    private fun startBackgroundService() {
        val serviceIntent = Intent(this, CryDetectionService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(serviceIntent)
        } else {
            startService(serviceIntent)
        }
        Log.d(TAG, "Background service started")
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
        
        // Load saved history
        loadSavedHistory()
        
        updateHistoryVisibility()
    }
    
    /**
     * Load history from persistent storage
     */
    private fun loadSavedHistory() {
        val savedHistory = HistoryManager.loadHistory(this)
        if (savedHistory.isNotEmpty()) {
            historyAdapter.setHistoryList(savedHistory)
            drawerHistoryAdapter.setHistoryList(savedHistory)
        }
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
        
        // Save to persistent storage
        HistoryManager.saveHistory(this, historyAdapter.getHistoryList())
        
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
            binding.actionButtonText.text = "Dinlemeyi Durdur"
            saveListeningState(true)
            // Service başlatma - MainActivity kendi tespitini yapacak
            // Service sadece uygulama arka plandayken çalışacak (onPause'da başlatılır)
            startListening()
        } else {
            // Pause
            isPaused = true
            binding.actionButtonText.text = "Dinlemeye Devam Et"
            saveListeningState(false)
            stopBackgroundService()
            stopListening()
            updateStatus("Duraklatıldı", "Dinleme duraklatıldı.\nDevam etmek için Devam'a tıklayın.")
        }
    }
    
    /**
     * Save listening state to SharedPreferences
     */
    private fun saveListeningState(enabled: Boolean) {
        getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .edit()
            .putBoolean(PREF_LISTENING_ENABLED, enabled)
            .apply()
    }
    
    /**
     * Load listening state from SharedPreferences
     */
    private fun loadListeningState(): Boolean {
        return getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
            .getBoolean(PREF_LISTENING_ENABLED, true) // Default: enabled
    }
    
    /**
     * Stop background detection service
     */
    private fun stopBackgroundService() {
        val serviceIntent = Intent(this, CryDetectionService::class.java)
        stopService(serviceIntent)
        Log.d(TAG, "Background service stopped")
    }
    
    private fun reconnectArduino() {
        updateStatus("Bağlanıyor...", "Arduino aranıyor...")
        lifecycleScope.launch(Dispatchers.IO) {
            arduinoSerial?.disconnect()
            delay(500)
            val connected = arduinoSerial?.scanAndConnect() == true
            withContext(Dispatchers.Main) {
                if (connected) {
                    Toast.makeText(this@MainActivity, "Arduino Bağlandı", Toast.LENGTH_SHORT).show()
                } else {
                    Toast.makeText(this@MainActivity, "Arduino Bulunamadı", Toast.LENGTH_SHORT).show()
                }
                if (!isPaused) {
                    updateStatus("Dinleniyor...", "Arka planda dinleme aktif.\nOrtam sessiz.")
                }
            }
        }
    }
    
    override fun onResume() {
        super.onResume()
        
        // MainActivity açıkken Service'i durdur - çift tespit olmasın
        stopBackgroundService()
        
        // Arduino bağlantı kontrolü
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                arduinoSerial?.scanAndConnect()
            } catch (e: Exception) {
                Log.e(TAG, "Arduino scan error", e)
            }
        }
        
        // Eğer duraklatılmamışsa dinlemeye başla
        if (!isPaused) {
            startListening()
        }
    }
    
    override fun onPause() {
        super.onPause()
        stopListening()
        stopPulseAnimation()
        
        // Uygulama arka plana gidince Service'i başlat (eğer duraklatılmamışsa)
        if (!isPaused) {
            startBackgroundService()
        }
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
                    Toast.makeText(context, "USB Cihazı Algılandı", Toast.LENGTH_SHORT).show()
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
                    Toast.makeText(context, "USB Cihazı Kaldırıldı", Toast.LENGTH_SHORT).show()
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
        updateStatus("Yükleniyor...", "Modeller başlatılıyor...")
        
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
                    updateStatus("Hazır", "Modeller başarıyla yüklendi.")
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
        // First check notification permission for Android 13+
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(
                    this, Manifest.permission.POST_NOTIFICATIONS
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                notificationPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
                return
            }
        }
        
        // Check if listening was previously disabled by user
        val wasListeningEnabled = loadListeningState()
        if (!wasListeningEnabled) {
            isPaused = true
            binding.actionButtonText.text = "Dinlemeye Devam Et"
            updateStatus("Duraklatıldı", "Dinleme duraklatıldı.\nDevam etmek için Devam'a tıklayın.")
            return
        }
        
        // Then check audio permission
        when {
            ContextCompat.checkSelfPermission(
                this, Manifest.permission.RECORD_AUDIO
            ) == PackageManager.PERMISSION_GRANTED -> {
                // Service başlatma - MainActivity açıkken kendi tespitini yapacak
                // Service sadece uygulama arka plandayken çalışacak (onPause'da başlatılır)
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
        updateStatus("Dinleniyor...", "Arka planda dinleme aktif.\nOrtam sessiz.")
        startPulseAnimation()
        
        // Update status badge
        binding.statusBadgeText.text = "CANLI"
        binding.statusDot.setBackgroundResource(R.drawable.circle_small)
        
        lifecycleScope.launch(Dispatchers.IO) {
            arduinoSerial?.sendInfo("Dinleniyor...", "Bebek bekleniyor")
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
        
        binding.statusBadgeText.text = "DURDURULDU"
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
                            updateStatus("Dinleniyor...", "Arka planda dinleme aktif.\nOrtam sessiz.")
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
                    updateStatus("Analiz ediliyor...", "Bebek ağlaması tespit edildi!\nSebep analiz ediliyor...")
                    stopPulseAnimation()
                    binding.statusBadgeText.text = "TESPİT"
                }
            }
            
            detectionResults.add(result)
            
            val elapsedSeconds = (System.currentTimeMillis() - detectionStartTime) / 1000
            withContext(Dispatchers.Main) {
                updateStatus("Analiz ediliyor...", "İşleniyor... (${elapsedSeconds}sn / 5sn)")
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
                        
                        // Ninni+Oyuncak'ı arka planda başlat (dialog'u bekleme)
                        if (finalResult.predictedClass in listOf("tired", "discomfort")) {
                            Log.d(TAG, "Baby tired/uncomfortable - Starting soothe mode in background...")
                            // Arka planda çalıştır, dialog'u bloklamaz
                            lifecycleScope.launch(Dispatchers.IO) {
                                arduinoSerial?.sendInfo("Ninni+Oyuncak", "Bebek sakinles")
                                arduinoSerial?.playSoothe()
                                // Ninni bitene kadar bekle
                                delay(35000)
                                Log.d(TAG, "Soothe mode finished")
                            }
                        }
                        
                        // Dialog'u HEMEN göster (ninni beklemeden)
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
