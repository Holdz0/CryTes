package com.ciona.babycry.audio

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlinx.coroutines.*
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlin.math.sqrt

/**
 * AudioCapture - Mikrofon kaydı ve ses verisi yönetimi
 * 
 * YAMNet için 16kHz mono PCM ses kaydı yapar.
 * Ring buffer ile sürekli 5 saniyelik veri tutar.
 */
class AudioCapture {
    
    companion object {
        const val SAMPLE_RATE = 16000          // YAMNet zorunluluğu
        const val CHANNEL_CONFIG = AudioFormat.CHANNEL_IN_MONO
        const val AUDIO_FORMAT = AudioFormat.ENCODING_PCM_16BIT
        const val BUFFER_DURATION_SEC = 5      // 5 saniyelik buffer
        const val CHUNK_DURATION_MS = 500      // 0.5 saniyelik okumalar
        const val RMS_THRESHOLD = 0.005f       // Sessizlik eşiği
    }
    
    private var audioRecord: AudioRecord? = null
    private var recordingJob: Job? = null
    
    // Ring Buffer (5 saniyelik)
    private val bufferSize = SAMPLE_RATE * BUFFER_DURATION_SEC
    private val ringBuffer = FloatArray(bufferSize)
    private var writePosition = 0
    private var bufferFilled = false
    
    // Audio event flow
    private val _audioEvents = MutableSharedFlow<AudioEvent>()
    val audioEvents: SharedFlow<AudioEvent> = _audioEvents
    
    // Chunk size (0.5 saniye = 8000 sample)
    private val chunkSamples = (SAMPLE_RATE * CHUNK_DURATION_MS) / 1000
    
    /**
     * Kaydı başlat
     */
    @SuppressLint("MissingPermission")
    fun start(): Boolean {
        val minBufferSize = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, CHANNEL_CONFIG, AUDIO_FORMAT
        )
        
        if (minBufferSize == AudioRecord.ERROR || minBufferSize == AudioRecord.ERROR_BAD_VALUE) {
            return false
        }
        
        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            SAMPLE_RATE,
            CHANNEL_CONFIG,
            AUDIO_FORMAT,
            minBufferSize * 2
        )
        
        if (audioRecord?.state != AudioRecord.STATE_INITIALIZED) {
            audioRecord?.release()
            audioRecord = null
            return false
        }
        
        audioRecord?.startRecording()
        
        recordingJob = CoroutineScope(Dispatchers.IO).launch {
            val shortBuffer = ShortArray(chunkSamples)
            
            while (isActive) {
                val readCount = audioRecord?.read(shortBuffer, 0, chunkSamples) ?: 0
                
                if (readCount > 0) {
                    // Short -> Float dönüşümü [-1.0, 1.0]
                    val floatChunk = FloatArray(readCount) { i ->
                        shortBuffer[i] / 32768f
                    }
                    
                    // Ring buffer'a ekle
                    addToBuffer(floatChunk)
                    
                    // RMS hesapla
                    val rms = calculateRMS(floatChunk)
                    
                    // Event gönder
                    if (rms >= RMS_THRESHOLD && bufferFilled) {
                        _audioEvents.emit(AudioEvent.AudioReady(getFullBuffer(), rms))
                    } else {
                        _audioEvents.emit(AudioEvent.Silence(rms))
                    }
                }
            }
        }
        
        return true
    }
    
    /**
     * Kaydı durdur
     */
    fun stop() {
        recordingJob?.cancel()
        recordingJob = null
        
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        
        // Buffer'ı temizle
        writePosition = 0
        bufferFilled = false
    }
    
    /**
     * Ring buffer'a veri ekle
     */
    private fun addToBuffer(chunk: FloatArray) {
        for (sample in chunk) {
            ringBuffer[writePosition] = sample
            writePosition = (writePosition + 1) % bufferSize
            
            if (writePosition == 0) {
                bufferFilled = true
            }
        }
    }
    
    /**
     * Tam buffer'ı al (5 saniyelik veri)
     */
    private fun getFullBuffer(): FloatArray {
        val result = FloatArray(bufferSize)
        
        if (bufferFilled) {
            // Ring buffer'ı düzgün sırayla oku
            for (i in 0 until bufferSize) {
                val readPos = (writePosition + i) % bufferSize
                result[i] = ringBuffer[readPos]
            }
        } else {
            // Buffer henüz dolmadı, mevcut veriyi kopyala
            System.arraycopy(ringBuffer, 0, result, 0, writePosition)
        }
        
        return result
    }
    
    /**
     * Buffer'ı temizle (tespit sonrası)
     */
    fun clearBuffer() {
        writePosition = 0
        bufferFilled = false
        ringBuffer.fill(0f)
    }
    
    /**
     * RMS (Root Mean Square) hesapla - Ses enerjisi
     */
    private fun calculateRMS(samples: FloatArray): Float {
        if (samples.isEmpty()) return 0f
        var sum = 0f
        for (sample in samples) {
            sum += sample * sample
        }
        return sqrt(sum / samples.size)
    }
}

/**
 * Audio durumu event'leri
 */
sealed class AudioEvent {
    data class AudioReady(val samples: FloatArray, val rms: Float) : AudioEvent()
    data class Silence(val rms: Float) : AudioEvent()
}
