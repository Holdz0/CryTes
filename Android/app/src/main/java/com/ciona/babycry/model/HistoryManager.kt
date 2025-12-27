package com.ciona.babycry.model

import android.content.Context
import android.content.SharedPreferences
import org.json.JSONArray
import org.json.JSONObject

/**
 * HistoryManager - Geçmiş kayıtlarını kalıcı olarak saklar
 */
object HistoryManager {
    
    private const val PREFS_NAME = "crytes_history"
    private const val KEY_HISTORY = "cry_history"
    private const val MAX_HISTORY_SIZE = 100
    
    private fun getPrefs(context: Context): SharedPreferences {
        return context.getSharedPreferences(PREFS_NAME, Context.MODE_PRIVATE)
    }
    
    /**
     * Geçmişi kaydet
     */
    fun saveHistory(context: Context, historyList: List<CryHistory>) {
        val jsonArray = JSONArray()
        
        // Son MAX_HISTORY_SIZE kadar kaydet
        val listToSave = historyList.take(MAX_HISTORY_SIZE)
        
        for (history in listToSave) {
            val jsonObject = JSONObject().apply {
                put("id", history.id)
                put("cryType", history.cryType)
                put("cryLabel", history.cryLabel)
                put("emoji", history.emoji)
                put("confidence", history.confidence.toDouble())
                put("timestamp", history.timestamp)
            }
            jsonArray.put(jsonObject)
        }
        
        getPrefs(context).edit()
            .putString(KEY_HISTORY, jsonArray.toString())
            .apply()
    }
    
    /**
     * Geçmişi yükle
     */
    fun loadHistory(context: Context): List<CryHistory> {
        val jsonString = getPrefs(context).getString(KEY_HISTORY, null) ?: return emptyList()
        
        return try {
            val jsonArray = JSONArray(jsonString)
            val historyList = mutableListOf<CryHistory>()
            
            for (i in 0 until jsonArray.length()) {
                val jsonObject = jsonArray.getJSONObject(i)
                val history = CryHistory(
                    id = jsonObject.getLong("id"),
                    cryType = jsonObject.getString("cryType"),
                    cryLabel = jsonObject.getString("cryLabel"),
                    emoji = jsonObject.getString("emoji"),
                    confidence = jsonObject.getDouble("confidence").toFloat(),
                    timestamp = jsonObject.getLong("timestamp")
                )
                historyList.add(history)
            }
            
            historyList
        } catch (e: Exception) {
            emptyList()
        }
    }
    
    /**
     * Geçmişe yeni kayıt ekle ve kaydet
     */
    fun addAndSave(context: Context, history: CryHistory, currentList: List<CryHistory>): List<CryHistory> {
        val newList = mutableListOf(history)
        newList.addAll(currentList)
        saveHistory(context, newList)
        return newList
    }
    
    /**
     * Geçmişi temizle
     */
    fun clearHistory(context: Context) {
        getPrefs(context).edit()
            .remove(KEY_HISTORY)
            .apply()
    }
}
