package com.ciona.babycry

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.drawable.GradientDrawable
import android.os.Bundle
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
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.collectLatest

/**
 * MainActivity - Crytes Ana Ekranı
 * 
 * Bebek ağlaması algılama ve sınıflandırma uygulamasının ana activity'si.
 * Tüm modülleri (Audio, YAMNet, Classifier, Arduino) koordine eder.
 */
class MainActivity : AppCompatActivity() {
    
    private lateinit var binding: ActivityMainBinding
    
    // Modüller
    private var audioCapture: AudioCapture? = null
    private var yamnetProcessor: YamnetProcessor? = null
    private var cryClassifier: CryClassifier? = null
    private var arduinoSerial: ArduinoSerial? = null
    
    // Durum
    private var isListening = false
    private var processingJob: Job? = null
    
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
    }
    
    override fun onResume() {
        super.onResume()
        // Arduino bağlantısını kontrol et
        arduinoSerial?.scanAndConnect()
        observeArduinoConnection()
    }
    
    override fun onPause() {
        super.onPause()
        stopListening()
    }
    
    override fun onDestroy() {
        super.onDestroy()
        releaseModules()
    }
    
    /**
     * Modülleri başlat
     */
    private fun initializeModules() {
        updateStatus("Modeller yükleniyor...", StatusColor.PROCESSING)
        
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // ML modelleri yükle
                yamnetProcessor = YamnetProcessor(this@MainActivity)
                cryClassifier = CryClassifier(this@MainActivity)
                
                // Audio modülü
                audioCapture = AudioCapture()
                
                // Arduino modülü
                arduinoSerial = ArduinoSerial(this@MainActivity)
                
                withContext(Dispatchers.Main) {
                    updateStatus("Hazır", StatusColor.SUCCESS)
                    checkPermissionAndStart()
                }
                
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    updateStatus("Model yükleme hatası: ${e.message}", StatusColor.ERROR)
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
            capture.audioEvents.collectLatest { event ->
                when (event) {
                    is AudioEvent.AudioReady -> {
                        processAudio(event.samples, event.rms)
                    }
                    is AudioEvent.Silence -> {
                        updateDebug("RMS: ${String.format("%.4f", event.rms)} (sessiz)")
                    }
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
     * Ses verisini işle
     */
    private suspend fun processAudio(samples: FloatArray, rms: Float) {
        val yamnet = yamnetProcessor ?: return
        val classifier = cryClassifier ?: return
        
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
        
        // Baby cry değilse atla
        if (!yamnetResult.isBabyCrying) {
            return
        }
        
        // Baby cry tespit edildi - sınıflandır
        withContext(Dispatchers.Main) {
            updateStatus("Bebek ağlaması tespit edildi!", StatusColor.DETECTED)
        }
        
        val result = withContext(Dispatchers.Default) {
            classifier.classify(yamnetResult.embedding)
        }
        
        withContext(Dispatchers.Main) {
            showResult(result)
            
            // Arduino'ya gönder
            arduinoSerial?.sendResult(result.getLcdText())
            
            // 3 saniye bekle ve buffer'ı temizle
            delay(3000)
            audioCapture?.clearBuffer()
            hideResult()
            updateStatus("Dinleniyor...", StatusColor.LISTENING)
        }
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
