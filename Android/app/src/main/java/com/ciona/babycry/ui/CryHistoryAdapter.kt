package com.ciona.babycry.ui

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.ciona.babycry.R
import com.ciona.babycry.model.CryHistory

/**
 * Ağlama geçmişi için RecyclerView adapter
 */
class CryHistoryAdapter(
    private val historyList: MutableList<CryHistory> = mutableListOf()
) : RecyclerView.Adapter<CryHistoryAdapter.ViewHolder>() {

    // Color mappings for different cry types
    private val typeColors = mapOf(
        "hungry" to R.drawable.overlay_orange_circle,
        "tired" to R.drawable.overlay_purple_circle,
        "discomfort" to R.drawable.overlay_blue_circle,
        "belly_pain" to R.drawable.overlay_red_circle,
        "burping" to R.drawable.overlay_purple_circle
    )

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val iconContainer: FrameLayout = view.findViewById(R.id.iconContainer)
        val icon: ImageView = view.findViewById(R.id.historyIcon)
        val title: TextView = view.findViewById(R.id.historyTitle)
        val subtitle: TextView = view.findViewById(R.id.historySubtitle)
        val time: TextView = view.findViewById(R.id.historyTime)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_cry_history, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val item = historyList[position]
        
        // Set title with emoji
        holder.title.text = "${item.emoji} ${item.cryLabel} Detected"
        
        // Set subtitle based on confidence
        val intensity = when {
            item.confidence >= 0.8f -> "High"
            item.confidence >= 0.5f -> "Medium"
            else -> "Low"
        }
        holder.subtitle.text = "$intensity intensity duration"
        
        // Set relative time
        holder.time.text = getRelativeTime(item.timestamp)
        
        // Set background color based on type
        val bgDrawable = typeColors[item.cryType] ?: R.drawable.overlay_orange_circle
        holder.iconContainer.setBackgroundResource(bgDrawable)
    }

    override fun getItemCount() = historyList.size

    fun addHistory(history: CryHistory) {
        historyList.add(0, history) // Most recent first
        notifyItemInserted(0)
    }

    fun isEmpty() = historyList.isEmpty()
    
    fun getHistoryList(): List<CryHistory> = historyList.toList()
    
    fun setHistoryList(list: List<CryHistory>) {
        historyList.clear()
        historyList.addAll(list)
        notifyDataSetChanged()
    }

    private fun getRelativeTime(timestamp: Long): String {
        val now = System.currentTimeMillis()
        val diff = now - timestamp
        
        val minutes = diff / (1000 * 60)
        val hours = diff / (1000 * 60 * 60)
        val days = diff / (1000 * 60 * 60 * 24)
        
        return when {
            minutes < 1 -> "now"
            minutes < 60 -> "${minutes}m ago"
            hours < 24 -> "${hours}h ago"
            days < 7 -> "${days}d ago"
            else -> "${days / 7}w ago"
        }
    }
}
