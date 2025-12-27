package com.ciona.babycry.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.content.Context
import android.content.Intent
import android.os.Build
import androidx.core.app.NotificationCompat
import com.ciona.babycry.MainActivity
import com.ciona.babycry.R
import com.ciona.babycry.ml.ClassificationResult

/**
 * NotificationHelper - Bildirim kanalları ve bildirim oluşturma
 */
object NotificationHelper {
    
    const val CHANNEL_SERVICE = "crytes_service"
    const val CHANNEL_DETECTION = "crytes_detection"
    const val NOTIFICATION_ID_SERVICE = 1
    const val NOTIFICATION_ID_DETECTION = 2
    
    /**
     * Bildirim kanallarını oluştur (Android 8.0+)
     */
    fun createChannels(context: Context) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val notificationManager = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
            
            // Servis kanalı (sabit bildirim)
            val serviceChannel = NotificationChannel(
                CHANNEL_SERVICE,
                "Crytes Aktif",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "Arka planda dinleme aktif bildirimi"
                setShowBadge(false)
            }
            
            // Tespit kanalı (ağlama bildirimi)
            val detectionChannel = NotificationChannel(
                CHANNEL_DETECTION,
                "Ağlama Tespiti",
                NotificationManager.IMPORTANCE_HIGH
            ).apply {
                description = "Bebek ağlaması tespit edildiğinde bildirim"
                enableVibration(true)
                enableLights(true)
            }
            
            notificationManager.createNotificationChannel(serviceChannel)
            notificationManager.createNotificationChannel(detectionChannel)
        }
    }
    
    /**
     * Sabit servis bildirimi oluştur
     * "Crytes Aktif - Dinleniyor" + Durdur butonu
     */
    fun createServiceNotification(context: Context, isListening: Boolean): Notification {
        // Ana uygulama açma intent'i
        val mainIntent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP
        }
        val mainPendingIntent = PendingIntent.getActivity(
            context, 0, mainIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        
        // Durdur butonu intent'i
        val stopIntent = Intent(context, CryDetectionService::class.java).apply {
            action = CryDetectionService.ACTION_STOP
        }
        val stopPendingIntent = PendingIntent.getService(
            context, 1, stopIntent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        
        val statusText = if (isListening) "Dinleniyor..." else "Duraklatıldı"
        
        return NotificationCompat.Builder(context, CHANNEL_SERVICE)
            .setContentTitle("🎯 Crytes Aktif")
            .setContentText(statusText)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setOngoing(true)
            .setContentIntent(mainPendingIntent)
            .addAction(
                android.R.drawable.ic_media_pause,
                "Durdur",
                stopPendingIntent
            )
            .setPriority(NotificationCompat.PRIORITY_LOW)
            .setCategory(NotificationCompat.CATEGORY_SERVICE)
            .build()
    }
    
    /**
     * Ağlama tespit bildirimi oluştur
     * Tıklandığında doğrudan ilgili soruyu açar
     */
    fun createDetectionNotification(context: Context, result: ClassificationResult, secondBest: String?): Notification {
        // MainActivity'ye cry type bilgisiyle intent
        val intent = Intent(context, MainActivity::class.java).apply {
            flags = Intent.FLAG_ACTIVITY_SINGLE_TOP or Intent.FLAG_ACTIVITY_CLEAR_TOP
            putExtra("CRY_TYPE", result.predictedClass)
            putExtra("CRY_LABEL", result.predictedLabel)
            putExtra("CRY_EMOJI", result.emoji)
            putExtra("CRY_CONFIDENCE", result.confidence)
            putExtra("CRY_SECOND_BEST", secondBest)
        }
        
        val pendingIntent = PendingIntent.getActivity(
            context, 
            System.currentTimeMillis().toInt(), // Unique request code
            intent,
            PendingIntent.FLAG_IMMUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
        )
        
        val displayConfidence = adjustConfidenceForDisplay(result.confidence)
        val title = "${result.emoji} ${result.predictedLabel} Tespit Edildi!"
        val text = "%$displayConfidence güvenle tespit edildi"
        
        return NotificationCompat.Builder(context, CHANNEL_DETECTION)
            .setContentTitle(title)
            .setContentText(text)
            .setSmallIcon(R.drawable.ic_launcher_foreground)
            .setAutoCancel(true)
            .setContentIntent(pendingIntent)
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setDefaults(NotificationCompat.DEFAULT_ALL)
            .build()
    }
    
    /**
     * Güven oranını görüntüleme için ayarla
     * 23-50%: +25 eklenir
     * 50-60%: +10 eklenir
     * 60-70%: +5 eklenir
     * 70-100%: olduğu gibi kalır
     */
    fun adjustConfidenceForDisplay(confidence: Float): Int {
        val percent = (confidence * 100).toInt()
        
        val adjustment = when {
            percent < 23 -> 25  // Çok düşükse +25
            percent in 23..49 -> 25  // 23-50 arası +25
            percent in 50..59 -> 10  // 50-60 arası +10
            percent in 60..69 -> 5   // 60-70 arası +5
            else -> 0  // 70+ olduğu gibi
        }
        
        return minOf(percent + adjustment, 99)  // Max %99
    }
}
