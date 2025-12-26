package com.ciona.babycry.serial

import android.app.PendingIntent
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.hardware.usb.UsbDevice
import android.hardware.usb.UsbDeviceConnection
import android.hardware.usb.UsbManager
import android.os.Build
import com.hoho.android.usbserial.driver.UsbSerialDriver
import com.hoho.android.usbserial.driver.UsbSerialPort
import com.hoho.android.usbserial.driver.UsbSerialProber
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import java.io.IOException

/**
 * ArduinoSerial - USB OTG üzerinden Arduino iletişimi
 * 
 * USB Serial kütüphanesi kullanarak Arduino ile haberleşir.
 * Baud rate: 9600 (live_detection.py ile aynı)
 */
class ArduinoSerial(private val context: Context) {
    
    companion object {
        private const val ACTION_USB_PERMISSION = "com.ciona.babycry.USB_PERMISSION"
        private const val BAUD_RATE = 9600
        private const val DATA_BITS = 8
        private const val WRITE_TIMEOUT = 1000  // ms
    }
    
    private var usbManager: UsbManager = context.getSystemService(Context.USB_SERVICE) as UsbManager
    private var serialPort: UsbSerialPort? = null
    private var connection: UsbDeviceConnection? = null
    
    // Bağlantı durumu
    private val _isConnected = MutableStateFlow(false)
    val isConnected: StateFlow<Boolean> = _isConnected
    
    // Bağlı cihaz bilgisi
    private val _deviceName = MutableStateFlow<String?>(null)
    val deviceName: StateFlow<String?> = _deviceName
    
    // USB izin receiver
    private val usbPermissionReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            if (ACTION_USB_PERMISSION == intent.action) {
                synchronized(this) {
                    val device = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                        intent.getParcelableExtra(UsbManager.EXTRA_DEVICE, UsbDevice::class.java)
                    } else {
                        @Suppress("DEPRECATION")
                        intent.getParcelableExtra(UsbManager.EXTRA_DEVICE)
                    }
                    
                    if (intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)) {
                        device?.let { connectToDevice(it) }
                    }
                }
            }
        }
    }
    
    init {
        // USB izin receiver'ı kaydet
        val filter = IntentFilter(ACTION_USB_PERMISSION)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            context.registerReceiver(usbPermissionReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
        } else {
            context.registerReceiver(usbPermissionReceiver, filter)
        }
    }
    
    /**
     * Bağlı USB cihazlarını tara ve Arduino'ya bağlan
     */
    fun scanAndConnect(): Boolean {
        val availableDrivers = UsbSerialProber.getDefaultProber().findAllDrivers(usbManager)
        
        if (availableDrivers.isEmpty()) {
            _isConnected.value = false
            _deviceName.value = null
            return false
        }
        
        // İlk uygun cihazı al
        val driver = availableDrivers[0]
        val device = driver.device
        
        // İzin kontrolü
        if (!usbManager.hasPermission(device)) {
            val permissionIntent = PendingIntent.getBroadcast(
                context, 
                0, 
                Intent(ACTION_USB_PERMISSION),
                PendingIntent.FLAG_MUTABLE
            )
            usbManager.requestPermission(device, permissionIntent)
            return false
        }
        
        return connectToDevice(device)
    }
    
    /**
     * Belirli bir USB cihazına bağlan
     */
    private fun connectToDevice(device: UsbDevice): Boolean {
        try {
            val drivers = UsbSerialProber.getDefaultProber().findAllDrivers(usbManager)
            val driver = drivers.find { it.device == device } ?: return false
            
            connection = usbManager.openDevice(device)
            if (connection == null) {
                return false
            }
            
            serialPort = driver.ports[0]
            serialPort?.open(connection)
            serialPort?.setParameters(BAUD_RATE, DATA_BITS, UsbSerialPort.STOPBITS_1, UsbSerialPort.PARITY_NONE)
            
            _isConnected.value = true
            _deviceName.value = device.productName ?: "USB Serial Device"
            
            return true
            
        } catch (e: IOException) {
            e.printStackTrace()
            disconnect()
            return false
        }
    }
    
    /**
     * Arduino'ya mesaj gönder
     * 
     * @param message Gönderilecek metin (newline otomatik eklenir)
     */
    fun send(message: String): Boolean {
        val port = serialPort ?: return false
        
        return try {
            val data = "$message\n".toByteArray(Charsets.US_ASCII)
            port.write(data, WRITE_TIMEOUT)
            true
        } catch (e: IOException) {
            e.printStackTrace()
            false
        }
    }
    
    /**
     * Sınıflandırma sonucunu Arduino'ya gönder
     * 
     * @param lcdText LCD için formatlanmış metin
     */
    fun sendResult(lcdText: String): Boolean {
        return send(lcdText)
    }
    
    /**
     * Bağlantıyı kapat
     */
    fun disconnect() {
        try {
            serialPort?.close()
        } catch (e: IOException) {
            e.printStackTrace()
        }
        
        serialPort = null
        connection?.close()
        connection = null
        
        _isConnected.value = false
        _deviceName.value = null
    }
    
    /**
     * Kaynakları serbest bırak
     */
    fun release() {
        disconnect()
        try {
            context.unregisterReceiver(usbPermissionReceiver)
        } catch (e: IllegalArgumentException) {
            // Receiver zaten kayıtlı değil
        }
    }
}
