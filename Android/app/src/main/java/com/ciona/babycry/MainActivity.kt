package com.ciona.babycry

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.ciona.babycry.audio.AudioCapture
import com.ciona.babycry.audio.AudioEvent
import com.ciona.babycry.databinding.ActivityMainBinding
import com.ciona.babycry.ml.ClassificationResult
import com.ciona.babycry.ml.CryClassifier
import com.ciona.babycry.ml.YamnetProcessor
import com.ciona.babycry.serial.ArduinoSerial
import android.hardware.usb.UsbManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.widget.Toast
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.collectLatest

/**
 * MainActivity - Crytes Ana Ekranı
 */
class MainActivity : AppCompatActivity() {
    
    companion object {
        private const val TAG = "MainActivity"
        private const val DETECTION_WINDOW_MS = 5000L
        private const val SERIAL_SEND_INTERVAL = 1000L // 1 saniyede bir güncelle
    }
    
    private lateinit var binding: ActivityMainBinding
    
    // Modüller
    private var audioCapture: AudioCapture? = null
    private var yamnetProcessor: YamnetProcessor? = null
    private var cryClassifier: CryClassifier? = null
    private var arduinoSerial: ArduinoSerial? = null
    
    // Durum
    private var isListening = false
    private var processingJob: Job? = null
    private var isProcessing = false
    
    // Debouncing için tespit toplama
    private val detectionResults = mutableListOf<ClassificationResult>()
    private var detectionStartTime: Long = 0
    private var isCollectingDetections = false

    
    // Serial haberleşme hızı kontrolü
    private var lastSerialSendTime = 0L

    
    // İzin launcher
    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) {
            startListening()
        } else {
            updateStatus("Mikrofon izni gerekli", StatusColor.ERROR)
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // Modülleri başlat
        initializeModules()
        
        // Runtime USB tak-çıkar dinleyicisi
        registerUsbReceiver()
        
        // Manuel bağlanma butonu
        binding.arduinoStatus.setOnClickListener {
            updateStatus("Arduino aranıyor...", StatusColor.PROCESSING)
            lifecycleScope.launch(Dispatchers.IO) {
                arduinoSerial?.disconnect() // Önce temizle
                delay(500)
                val connected = arduinoSerial?.scanAndConnect() == true
                withContext(Dispatchers.Main) {
                    if (connected) {
                        Toast.makeText(this@MainActivity, "Arduino Bağlandı", Toast.LENGTH_SHORT).show()
                    } else {
                        Toast.makeText(this@MainActivity, "Arduino Bulunamadı", Toast.LENGTH_SHORT).show()
                        updateStatus("Dinleniyor...", StatusColor.LISTENING)
                    }
                }
            }
        }
    }
    
    override fun onResume() {
        super.onResume()
        // Arduino bağlantısını kontrol et (Background)
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
    }
    
    override fun onDestroy() {
        super.onDestroy()
        unregisterUsbReceiver()
        releaseModules()
    }
    
    // --- USB Receiver ---
    private val usbReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            when (intent.action) {
                UsbManager.ACTION_USB_DEVICE_ATTACHED -> {
                    Log.d(TAG, "USB Device Attached")
                    Toast.makeText(context, "USB Cihazı Algılandı", Toast.LENGTH_SHORT).show()
                    
                    // Ses sistemini yeniden başlat (Routing değişimi için)
                    stopListening()
                    lifecycleScope.launch(Dispatchers.IO) {
                        delay(1000) // Cihazın hazır olması için bekle
                        arduinoSerial?.scanAndConnect()
                        
                        withContext(Dispatchers.Main) {
                            startListening() // Sesi tekrar başlat
                        }
                    }
                }
                UsbManager.ACTION_USB_DEVICE_DETACHED -> {
                    Log.d(TAG, "USB Device Detached")
                    Toast.makeText(context, "USB Cihazı Çıkarıldı", Toast.LENGTH_SHORT).show()
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
    
    /**
     * Modülleri başlat
     */
    private fun initializeModules() {
        updateStatus("Modeller yükleniyor...", StatusColor.PROCESSING)
        
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
                
                // Observer'ı sadece bir kez başlat
                withContext(Dispatchers.Main) {
                    observeArduinoConnection()
                    updateStatus("Hazır", StatusColor.SUCCESS)
                    checkPermissionAndStart()
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Module initialization error", e)
                withContext(Dispatchers.Main) {
                    updateStatus("Hata: ${e.message?.take(50)}", StatusColor.ERROR)
                }
            }
        }
    }
    
    /**
     * İzin kontrolü ve dinlemeyi başlat
     */
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
    
    /**
     * Ses dinlemeyi başlat
     */
    private fun startListening() {
        if (isListening) return
        
        val capture = audioCapture ?: return
        
        if (!capture.start()) {
            updateStatus("Mikrofon başlatılamadı", StatusColor.ERROR)
            return
        }
        
        isListening = true
        updateStatus("Dinleniyor...", StatusColor.LISTENING)
        
        // Audio event'lerini dinle
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
                            updateDebug("RMS: ${String.format("%.4f", event.rms)} (sessiz)")
                        }
                        is AudioEvent.Error -> {
                            Log.e(TAG, "Audio Error: ${event.message}")
                            withContext(Dispatchers.Main) {
                                updateDebug("Mikrofon Hatası: ${event.message}")
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                Log.e(TAG, "Audio processing error", e)
                withContext(Dispatchers.Main) {
                    updateDebug("Hata: ${e.message}")
                }
            }
        }
    }
    
    /**
     * Ses dinlemeyi durdur
     */
    private fun stopListening() {
        isListening = false
        processingJob?.cancel()
        processingJob = null
        audioCapture?.stop()
    }
    
    /**
     * Ses verisini işle - Debouncing ile
     */
    private suspend fun processAudio(samples: FloatArray, rms: Float) {
        val yamnet = yamnetProcessor ?: return
        val classifier = cryClassifier ?: return
        
        isProcessing = true
        
        try {
            withContext(Dispatchers.Main) {
                updateDebug("RMS: ${String.format("%.4f", rms)} | İşleniyor...")
            }
            
            // YAMNet ile embedding ve baby cry kontrolü
            val yamnetResult = withContext(Dispatchers.Default) {
                yamnet.process(samples)
            }
            
            withContext(Dispatchers.Main) {
                updateDebug("RMS: ${String.format("%.4f", rms)} | ${yamnetResult.topClassName} (${String.format("%.1f", yamnetResult.topScore * 100)}%)")
            }
            
            // Baby cry değilse
            if (!yamnetResult.isBabyCrying) {
                // Eğer toplama modundaysak ve uzun süre baby cry gelmezse sıfırla
                if (isCollectingDetections) {
                    val elapsed = System.currentTimeMillis() - detectionStartTime
                    // 2 saniyeden fazla sessizlik varsa toplama iptal
                    if (elapsed > 2000 && detectionResults.isEmpty()) {
                        isCollectingDetections = false
                        withContext(Dispatchers.Main) {
                            updateStatus("Dinleniyor...", StatusColor.LISTENING)
                        }
                    }
                }
                
                // Diğer sesleri Arduino'ya gönder (Hız kontrollü)
                val now = System.currentTimeMillis()
                if (now - lastSerialSendTime >= SERIAL_SEND_INTERVAL) {
                    lastSerialSendTime = now
                    try {
                        // İngilizce etiketleri basitçe Türkçeye çevirip gönderelim veya olduğu gibi
                        // YAMNet sınıfları genelde İngilizce: "Speech", "Music", "Silence" vs.
                        val labelToSend = when(yamnetResult.topClassName) {
                            "Speech" -> "Konusma"
                            "Silence" -> "Sessizlik"
                            "Music" -> "Muzik"
                            else -> yamnetResult.topClassName
                        }
                        
                        // IO thread'de gönder
                        arduinoSerial?.sendResult(labelToSend)
                    } catch (e: Exception) {
                        Log.e(TAG, "Non-cry serial error", e)
                    }
                }
                
                isProcessing = false
                return
            }
            
            // Baby cry tespit edildi!
            val result = withContext(Dispatchers.Default) {
                classifier.classify(yamnetResult.embedding)
            }
            
            // Toplama modunu başlat veya devam et
            if (!isCollectingDetections) {
                isCollectingDetections = true
                detectionStartTime = System.currentTimeMillis()
                detectionResults.clear()
                
                withContext(Dispatchers.Main) {
                    updateStatus("Bebek ağlaması analiz ediliyor...", StatusColor.DETECTED)
                }
            }
            
            // Sonucu listeye ekle
            detectionResults.add(result)
            Log.d(TAG, "Detection added: ${result.predictedLabel} (${detectionResults.size} samples)")
            
            // Anlık güncelleme göster
            val elapsedSeconds = (System.currentTimeMillis() - detectionStartTime) / 1000
            withContext(Dispatchers.Main) {
                updateStatus("Analiz ediliyor... (${elapsedSeconds}s / 5s)", StatusColor.DETECTED)
            }
            
            // 5 saniye doldu mu?
            val elapsed = System.currentTimeMillis() - detectionStartTime
            if (elapsed >= DETECTION_WINDOW_MS && detectionResults.isNotEmpty()) {
                // Voting ile final sonucu belirle
                val finalResult = computeFinalResult(detectionResults)
                
                Log.d(TAG, "Final decision after ${detectionResults.size} samples: ${finalResult.predictedLabel}")
                
                withContext(Dispatchers.Main) {
                    showResult(finalResult)
                }
                
                // Arduino'ya gönder (Suspend - IO ve Main dışı)
                try {
                    arduinoSerial?.sendResult(finalResult.getLcdText())
                } catch (e: Exception) {
                    Log.e(TAG, "Arduino send error", e)
                }
                
                // Sıfırla ve bekle
                isCollectingDetections = false
                detectionResults.clear()
                
                // 5 saniye bekle ve buffer'ı temizle
                delay(5000)
                audioCapture?.clearBuffer()
                
                withContext(Dispatchers.Main) {
                    hideResult()
                    updateStatus("Dinleniyor...", StatusColor.LISTENING)
                }
            }
            
        } catch (e: Exception) {
            Log.e(TAG, "Process audio error", e)
            withContext(Dispatchers.Main) {
                updateDebug("İşleme hatası: ${e.message?.take(30)}")
            }
        } finally {
            isProcessing = false
        }
    }
    
    /**
     * Voting sistemi ile final sonucu hesapla
     */
    private fun computeFinalResult(results: List<ClassificationResult>): ClassificationResult {
        if (results.isEmpty()) {
            return ClassificationResult(
                predictedClass = "unknown",
                predictedLabel = "Bilinmiyor",
                emoji = "❓",
                confidence = 0f,
                isConfident = false,
                allProbabilities = emptyMap()
            )
        }
        
        // Her sınıf için oy topla (confidence ağırlıklı)
        val votes = mutableMapOf<String, Float>()
        val probSums = mutableMapOf<String, MutableMap<String, Float>>()
        
        for (result in results) {
            val cls = result.predictedClass
            votes[cls] = (votes[cls] ?: 0f) + result.confidence
            
            // Probability toplamlarını da tut
            if (!probSums.containsKey(cls)) {
                probSums[cls] = mutableMapOf()
            }
            for ((k, v) in result.allProbabilities) {
                probSums[cls]!![k] = (probSums[cls]!![k] ?: 0f) + v
            }
        }
        
        // En yüksek oyu alan sınıfı bul
        val winningClass = votes.maxByOrNull { it.value }?.key ?: results.last().predictedClass
        
        // O sınıfa ait sonuçların ortalamasını al
        val classResults = results.filter { it.predictedClass == winningClass }
        val avgConfidence = classResults.map { it.confidence }.average().toFloat()
        
        // Ortalama probabilities
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
    
    /**
     * Sonucu göster
     */
    private fun showResult(result: ClassificationResult) {
        binding.resultCard.visibility = View.VISIBLE
        binding.resultEmoji.text = result.emoji
        binding.resultLabel.text = result.predictedLabel
        binding.resultConfidence.text = "Güven: %${String.format("%.1f", result.confidence * 100)}"
        
        // Güven rengi
        binding.resultConfidence.setTextColor(
            if (result.isConfident) getColor(R.color.success) else getColor(R.color.warning)
        )
        
        // Confidence bar'ları göster
        showConfidenceBars(result.allProbabilities, result.predictedClass)
    }
    
    /**
     * Sonucu gizle
     */
    private fun hideResult() {
        binding.resultCard.visibility = View.GONE
        binding.confidenceBarsContainer.removeAllViews()
    }
    
    /**
     * Tüm sınıfların güven bar'larını göster
     */
    private fun showConfidenceBars(probabilities: Map<String, Float>, selectedClass: String) {
        binding.confidenceBarsContainer.removeAllViews()
        
        if (probabilities.isEmpty()) return
        
        // Olasılığa göre sırala
        val sorted = probabilities.entries.sortedByDescending { it.value }
        
        for ((className, prob) in sorted) {
            val container = LinearLayout(this).apply {
                orientation = LinearLayout.HORIZONTAL
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { 
                    topMargin = 8 
                }
            }
            
            // Etiket
            val label = TextView(this).apply {
                text = "${CryClassifier.CLASS_EMOJIS[className] ?: ""} ${CryClassifier.CLASS_LABELS_TR[className] ?: className}"
                textSize = 12f
                setTextColor(if (className == selectedClass) getColor(R.color.primary) else getColor(R.color.on_surface))
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
            }
            
            // Progress bar
            val progress = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal).apply {
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                max = 100
                this.progress = (prob * 100).toInt()
            }
            
            // Yüzde
            val percent = TextView(this).apply {
                text = "${String.format("%.1f", prob * 100)}%"
                textSize = 12f
                setTextColor(getColor(R.color.on_surface))
                layoutParams = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                ).apply { marginStart = 8 }
            }
            
            container.addView(label)
            container.addView(progress)
            container.addView(percent)
            
            binding.confidenceBarsContainer.addView(container)
        }
    }
    
    /**
     * Durum mesajını güncelle
     */
    private fun updateStatus(message: String, color: StatusColor) {
        binding.statusText.text = message
        
        val indicatorColor = when (color) {
            StatusColor.LISTENING -> getColor(R.color.status_listening)
            StatusColor.PROCESSING -> getColor(R.color.status_processing)
            StatusColor.DETECTED -> getColor(R.color.status_detected)
            StatusColor.SUCCESS -> getColor(R.color.success)
            StatusColor.ERROR -> getColor(R.color.error)
        }
        
        (binding.statusIndicator.background as? GradientDrawable)?.setColor(indicatorColor)
            ?: binding.statusIndicator.setBackgroundColor(indicatorColor)
    }
    
    /**
     * Debug bilgisini güncelle
     */
    private fun updateDebug(text: String) {
        binding.debugText.text = text
    }
    
    /**
     * Arduino bağlantı durumunu gözlemle
     */
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
    
    /**
     * Kaynakları serbest bırak
     */
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
    
    /**
     * Durum renkleri
     */
    enum class StatusColor {
        LISTENING, PROCESSING, DETECTED, SUCCESS, ERROR
    }
}
