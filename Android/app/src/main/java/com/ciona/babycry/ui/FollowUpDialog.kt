package com.ciona.babycry.ui

import android.app.Dialog
import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.Window
import com.ciona.babycry.databinding.DialogFollowUpBinding
import com.ciona.babycry.ml.ClassificationResult

/**
 * Ebeveyn Takip Sorusu Diyaloğu
 */
class FollowUpDialog(
    context: Context,
    private val result: ClassificationResult,
    private val sensorData: String?,
    private val secondBestLabel: String?,
    private val onComplete: () -> Unit
) : Dialog(context) {

    private lateinit var binding: DialogFollowUpBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        binding = DialogFollowUpBinding.inflate(LayoutInflater.from(context))
        setContentView(binding.root)

        // Engellenemez yapalım, cevap verilmeli
        setCancelable(false)
        setCanceledOnTouchOutside(false)

        setupUI()
    }

    private fun setupUI() {
        // Başlık ve İkon
        binding.textTitle.text = "${result.emoji} ${result.predictedLabel} Tespit Edildi"
        
        // Sensör Bilgisi
        if (!sensorData.isNullOrEmpty()) {
            binding.textSensorInfo.text = sensorData
            binding.textSensorInfo.visibility = View.VISIBLE
        } else {
            binding.textSensorInfo.visibility = View.GONE
        }

        // Soru Mantığı
        val question = getQuestionForLabel(result.predictedClass)
        binding.textQuestion.text = question

        // Butonlar
        binding.btnYes.setOnClickListener {
            showAdvice(true)
        }

        binding.btnNo.setOnClickListener {
            showAdvice(false)
        }
        
        // Tamam butonu (başlangıçta gizli)
        binding.btnOk.setOnClickListener {
            dismiss()
            onComplete()
        }
    }

    private fun getQuestionForLabel(label: String): String {
        return when (label) {
            "hungry" -> "Bebek son 2 saat içerisinde yemek yedi mi?"
            "discomfort" -> "Bebeğin altı son 4 saat içerisinde temizlendi mi?"
            "tired" -> "Bebek bugün toplam 12 saat uyudu mu?"
            "burping" -> "Bebek gazını çıkarabildi mi?"
            "belly_pain" -> "Bebek son öğünden sonra rahatsızlandı mı?"
            else -> "Bebeğinizde bu durumu gözlemliyor musunuz?"
        }
    }

    private fun showAdvice(isYes: Boolean) {
        // Cevap butonlarını gizle
        binding.layoutButtons.visibility = View.GONE
        
        // Öneri kısmını göster
        binding.layoutAdvice.visibility = View.VISIBLE
        binding.textAdvice.text = getAdvice(result.predictedClass, isYes)
        
        // Eğer aç ve Hayır dediyse, hungry tutorial butonunu göster
        if (result.predictedClass == "hungry" && !isYes) {
            binding.btnTutorial.visibility = View.VISIBLE
            binding.btnTutorial.text = "📖 Besleme Rehberini Gör"
            binding.btnTutorial.setOnClickListener {
                HungryTutorialDialog(context) {
                    dismiss()
                    onComplete()
                }.show()
            }
        }
        
        // Eğer gaz/geğirme ve Hayır dediyse, burping tutorial butonunu göster
        if (result.predictedClass == "burping" && !isYes) {
            binding.btnTutorial.visibility = View.VISIBLE
            binding.btnTutorial.text = "📖 Gaz Çıkarma Rehberini Gör"
            binding.btnTutorial.setOnClickListener {
                BurpingTutorialDialog(context) {
                    dismiss()
                    onComplete()
                }.show()
            }
        }
        
        // Eğer yorgunluk ve Hayır dediyse, tired tutorial butonunu göster
        if (result.predictedClass == "tired" && !isYes) {
            binding.btnTutorial.visibility = View.VISIBLE
            binding.btnTutorial.text = "📖 Uyku Rehberini Gör"
            binding.btnTutorial.setOnClickListener {
                TiredTutorialDialog(context) {
                    dismiss()
                    onComplete()
                }.show()
            }
        }
    }

    private fun getAdvice(label: String, isYes: Boolean): String {
        val otherReason = secondBestLabel ?: "başka bir sebep"
        
        return when (label) {
            "hungry" -> if (isYes) 
                "💡 ÖNERİ: Bebek yakın zamanda yemek yediği için, ağlamanın sebebi $otherReason olabilir."
            else 
                "🍼 SONUÇ: Bebeğiniz aç! Lütfen bebeğinizi besleyin."

            "discomfort" -> if (isYes) 
                "💡 ÖNERİ: Bebeğin altı temiz olduğu için, ağlamanın sebebi $otherReason olabilir."
            else 
                "🧷 SONUÇ: Bebeğinizin altını temizlemeniz gerekiyor!"

            "tired" -> if (isYes) 
                "💡 ÖNERİ: Bebek yeterli uyku almış görünüyor, ağlamanın sebebi $otherReason olabilir."
            else 
                "🛏️ SONUÇ: Bebeğinizin uyuması gerekiyor!"

            "burping" -> if (isYes) 
                "💡 ÖNERİ: Bebek gazını çıkarmış görünüyor, ağlamanın sebebi $otherReason olabilir."
            else 
                "💨 SONUÇ: Bebeğinizin gazını çıkartması gerekiyor!"

            "belly_pain" -> if (isYes) 
                "⚠️ SONUÇ: Bebek yemekten sonra rahatsızlanmış olabilir. Gaz veya hazımsızlık olabilir."
            else 
                "💡 ÖNERİ: Karın ağrısının başka bir sebebi olabilir veya $otherReason durumu söz konusu olabilir."

            else -> "ℹ️ Bilgi: Bebeğinizi gözlemlemeye devam edin."
        }
    }
}
