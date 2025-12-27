package com.ciona.babycry.model

import java.text.SimpleDateFormat
import java.util.*

/**
 * Bebek ağlama geçmişi kaydı
 */
data class CryHistory(
    val id: Long = System.currentTimeMillis(),
    val cryType: String,          // "hunger", "tired", etc.
    val cryLabel: String,         // "Açlık", "Yorgunluk", etc.
    val emoji: String,            // "🍼", "😴", etc.
    val confidence: Float,        // 0.0 - 1.0
    val timestamp: Long = System.currentTimeMillis()
) {
    val formattedDateTime: String
        get() {
            val sdf = SimpleDateFormat("dd MMM yyyy, HH:mm", Locale("tr", "TR"))
            return sdf.format(Date(timestamp))
        }
    
    val formattedDate: String
        get() {
            val sdf = SimpleDateFormat("dd MMM", Locale("tr", "TR"))
            return sdf.format(Date(timestamp))
        }
    
    val formattedTime: String
        get() {
            val sdf = SimpleDateFormat("HH:mm", Locale("tr", "TR"))
            return sdf.format(Date(timestamp))
        }
}
