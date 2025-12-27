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
import android.util.Log
import com.hoho.android.usbserial.driver.UsbSerialPort
import com.hoho.android.usbserial.driver.UsbSerialProber
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import java.io.IOException

/**
 * ArduinoSerial - USB OTG üzerinden Arduino iletişimi
 */
class ArduinoSerial(private val context: Context) {
    
    companion object {
        private const val TAG = "ArduinoSerial"
        private const val ACTION_USB_PERMISSION = "com.ciona.babycry.USB_PERMISSION"
        private const val BAUD_RATE = 9600
        private const val DATA_BITS = 8
        private const val WRITE_TIMEOUT = 1000
    }
    
    private var usbManager: UsbManager = context.getSystemService(Context.USB_SERVICE) as UsbManager
    private var serialPort: UsbSerialPort? = null
    private var connection: UsbDeviceConnection? = null
    
    private val _isConnected = MutableStateFlow(false)
    val isConnected: StateFlow<Boolean> = _isConnected
    
    private val _deviceName = MutableStateFlow<String?>(null)
    val deviceName: StateFlow<String?> = _deviceName
    
    private var receiverRegistered = false
    
    private val usbPermissionReceiver = object : BroadcastReceiver() {
        override fun onReceive(context: Context, intent: Intent) {
            Log.d(TAG, "USB permission receiver triggered: ${intent.action}")
            
            if (ACTION_USB_PERMISSION == intent.action) {
                synchronized(this) {
                    val device = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                        intent.getParcelableExtra(UsbManager.EXTRA_DEVICE, UsbDevice::class.java)
                    } else {
                        @Suppress("DEPRECATION")
                        intent.getParcelableExtra(UsbManager.EXTRA_DEVICE)
                    }
                    
                    val granted = intent.getBooleanExtra(UsbManager.EXTRA_PERMISSION_GRANTED, false)
                    Log.d(TAG, "USB permission granted: $granted for device: ${device?.productName}")
                    
                    if (granted && device != null) {
                        // BroadcastReceiver main thread'de çalışır, bağlantıyı background'a atmalıyız
                        // Ancak connectToDevice şu an sync, context/scope lazım.
                        // Şimdilik basitçe çağırıyoruz, ama idealde coroutine scope lazım.
                        // connectToDevice içinde ağır işlem yoksa sorun olmaz.
                        // openDevice ve open() çağrıları senkron ama genelde hızlıdır.
                        connectToDevice(device)
                    }
                }
            }
        }
    }
    
    init {
        registerReceiver()
    }
    
    private fun registerReceiver() {
        if (receiverRegistered) return
        
        try {
            val filter = IntentFilter(ACTION_USB_PERMISSION)
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
                context.registerReceiver(usbPermissionReceiver, filter, Context.RECEIVER_NOT_EXPORTED)
            } else {
                context.registerReceiver(usbPermissionReceiver, filter)
            }
            receiverRegistered = true
            Log.d(TAG, "USB receiver registered")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to register receiver", e)
        }
    }
    
    /**
     * Bağlı USB cihazlarını tara ve Arduino'ya bağlan
     */
    /**
     * Bağlı USB cihazlarını tara ve Arduino'ya bağlan (Background Thread)
     */
    suspend fun scanAndConnect(): Boolean = withContext(Dispatchers.IO) {
        Log.d(TAG, "Scanning for USB devices...")
        
        try {
            // Önce tüm USB cihazlarını listele
            val deviceList = usbManager.deviceList
            Log.d(TAG, "Found ${deviceList.size} USB devices")
            
            for ((name, device) in deviceList) {
                Log.d(TAG, "USB Device: $name, VendorId: ${device.vendorId}, ProductId: ${device.productId}, Name: ${device.productName}")
            }
            
            // Serial driver'ları bul
            val availableDrivers = UsbSerialProber.getDefaultProber().findAllDrivers(usbManager)
            Log.d(TAG, "Found ${availableDrivers.size} serial drivers")
            
            if (availableDrivers.isEmpty()) {
                _isConnected.value = false
                _deviceName.value = null
                return@withContext false
            }
            
            val driver = availableDrivers[0]
            val device = driver.device
            Log.d(TAG, "Using device: ${device.productName}, VendorId: ${device.vendorId}")
            
            // İzin kontrolü
            if (!usbManager.hasPermission(device)) {
                Log.d(TAG, "Requesting USB permission...")
                val flags = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
                    PendingIntent.FLAG_MUTABLE or PendingIntent.FLAG_UPDATE_CURRENT
                } else {
                    PendingIntent.FLAG_UPDATE_CURRENT
                }
                
                val permissionIntent = PendingIntent.getBroadcast(
                    context,
                    0,
                    Intent(ACTION_USB_PERMISSION).apply {
                        setPackage(context.packageName)
                    },
                    flags
                )
                usbManager.requestPermission(device, permissionIntent)
                return@withContext false
            }
            
            return@withContext connectToDevice(device)
            
        } catch (e: Exception) {
            Log.e(TAG, "Error scanning USB devices", e)
            return@withContext false
        }
    }
    
    private fun connectToDevice(device: UsbDevice): Boolean {
        Log.d(TAG, "Connecting to device: ${device.productName}")
        
        try {
            val drivers = UsbSerialProber.getDefaultProber().findAllDrivers(usbManager)
            val driver = drivers.find { it.device == device }
            
            if (driver == null) {
                Log.e(TAG, "No driver found for device")
                return false
            }
            
            connection = usbManager.openDevice(device)
            if (connection == null) {
                Log.e(TAG, "Failed to open device")
                return false
            }
            
            serialPort = driver.ports[0]
            serialPort?.open(connection)
            serialPort?.setParameters(BAUD_RATE, DATA_BITS, UsbSerialPort.STOPBITS_1, UsbSerialPort.PARITY_NONE)
            
            _isConnected.value = true
            _deviceName.value = device.productName ?: "USB Serial Device"
            
            Log.d(TAG, "Connected successfully to ${device.productName}")
            return true
            
        } catch (e: Exception) {
            Log.e(TAG, "Failed to connect", e)
            disconnect()
            return false
        }
    }
    
    suspend fun send(message: String): Boolean = withContext(Dispatchers.IO) {
        val port = serialPort ?: return@withContext false
        
        return@withContext try {
            val data = "$message\n".toByteArray(Charsets.US_ASCII)
            port.write(data, WRITE_TIMEOUT)
            Log.d(TAG, "Sent: $message")
            true
        } catch (e: IOException) {
            Log.e(TAG, "Send failed", e)
            false
        }
    }
    
    suspend fun sendResult(lcdText: String): Boolean {
        return send(lcdText)
    }
    
    fun disconnect() {
        try {
            serialPort?.close()
        } catch (e: IOException) {
            Log.e(TAG, "Close error", e)
        }
        
        serialPort = null
        connection?.close()
        connection = null
        
        _isConnected.value = false
        _deviceName.value = null
    }
    
    fun release() {
        disconnect()
        try {
            if (receiverRegistered) {
                context.unregisterReceiver(usbPermissionReceiver)
                receiverRegistered = false
            }
        } catch (e: IllegalArgumentException) {
            Log.w(TAG, "Receiver already unregistered")
        }
    }
}
