package com.ciona.babycry.ui

import android.app.Dialog
import android.content.Context
import android.graphics.Color
import android.graphics.drawable.ColorDrawable
import android.os.CountDownTimer
import android.view.LayoutInflater
import android.view.Window
import android.widget.Button
import android.widget.TextView
import com.ciona.babycry.R
import com.ciona.babycry.serial.ArduinoSerial
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

/**
 * Etkinlikler Dialog - Ninni ve Işıkları manuel kontrol
 */
class ActivitiesDialog(
    context: Context,
    private val arduinoSerial: ArduinoSerial?,
    private val scope: CoroutineScope
) : Dialog(context) {

    companion object {
        private const val COOLDOWN_MS = 40_000L // 40 saniye
    }

    private var lullabyTimer: CountDownTimer? = null
    private var lightsTimer: CountDownTimer? = null
    
    private var isLullabyActive = false
    private var isLightsActive = false

    init {
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        window?.setBackgroundDrawable(ColorDrawable(Color.TRANSPARENT))
        
        val view = LayoutInflater.from(context).inflate(R.layout.dialog_activities, null)
        setContentView(view)
        
        val btnLullaby = view.findViewById<Button>(R.id.btnLullaby)
        val btnLights = view.findViewById<Button>(R.id.btnLights)
        val btnClose = view.findViewById<Button>(R.id.btnClose)
        val txtLullabyTimer = view.findViewById<TextView>(R.id.txtLullabyTimer)
        val txtLightsTimer = view.findViewById<TextView>(R.id.txtLightsTimer)
        
        // Ninni butonu
        btnLullaby.setOnClickListener {
            if (!isLullabyActive) {
                activateLullaby(btnLullaby, txtLullabyTimer)
            }
        }
        
        // Işıklar butonu
        btnLights.setOnClickListener {
            if (!isLightsActive) {
                activateLights(btnLights, txtLightsTimer)
            }
        }
        
        // Kapat butonu
        btnClose.setOnClickListener {
            dismiss()
        }
    }
    
    private fun activateLullaby(button: Button, timerText: TextView) {
        isLullabyActive = true
        button.isEnabled = false
        button.alpha = 0.5f
        button.text = "🎵 Ninni Çalıyor..."
        timerText.visibility = android.view.View.VISIBLE
        
        // Arduino'ya ninni + oyuncak komutu gönder (PLAY_SOOTHE)
        scope.launch(Dispatchers.IO) {
            arduinoSerial?.sendInfo("Ninni+Oyuncak", "Bebek sakinles")
            arduinoSerial?.playSoothe()  // Hem ninni hem oyuncak çalıştırır
        }
        
        // Cooldown timer
        lullabyTimer = object : CountDownTimer(COOLDOWN_MS, 1000) {
            override fun onTick(millisUntilFinished: Long) {
                val seconds = millisUntilFinished / 1000
                timerText.text = "Bekleme: ${seconds}s"
            }
            
            override fun onFinish() {
                isLullabyActive = false
                button.isEnabled = true
                button.alpha = 1f
                button.text = "🎵 Ninni Çal"
                timerText.visibility = android.view.View.GONE
            }
        }.start()
    }
    
    private fun activateLights(button: Button, timerText: TextView) {
        isLightsActive = true
        button.isEnabled = false
        button.alpha = 0.5f
        button.text = "💡 Işıklar Açık..."
        timerText.visibility = android.view.View.VISIBLE
        
        // Arduino'ya ışık komutu gönder (LED animasyonu)
        scope.launch(Dispatchers.IO) {
            arduinoSerial?.sendInfo("Isiklar Acik", "LED Animasyonu")
            arduinoSerial?.send("PLAY_LIGHTS")
        }
        
        // Cooldown timer
        lightsTimer = object : CountDownTimer(COOLDOWN_MS, 1000) {
            override fun onTick(millisUntilFinished: Long) {
                val seconds = millisUntilFinished / 1000
                timerText.text = "Bekleme: ${seconds}s"
            }
            
            override fun onFinish() {
                isLightsActive = false
                button.isEnabled = true
                button.alpha = 1f
                button.text = "💡 Işıkları Aç"
                timerText.visibility = android.view.View.GONE
            }
        }.start()
    }
    
    override fun dismiss() {
        lullabyTimer?.cancel()
        lightsTimer?.cancel()
        super.dismiss()
    }
}
