package com.ciona.babycry.ui

import android.app.Dialog
import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.Window
import android.view.WindowManager
import com.ciona.babycry.R
import com.ciona.babycry.databinding.DialogHungryTutorialBinding

/**
 * Aç Bebek Besleme Tutorial Diyaloğu
 * 7 adımlık görsel rehber
 */
class HungryTutorialDialog(
    context: Context,
    private val onComplete: () -> Unit
) : Dialog(context, android.R.style.Theme_Black_NoTitleBar_Fullscreen) {

    private lateinit var binding: DialogHungryTutorialBinding
    private var currentStep = 0

    // Tutorial görselleri (drawable resource IDs)
    private val tutorialImages = listOf(
        R.drawable.hungry1,
        R.drawable.hungry2,
        R.drawable.hungry3,
        R.drawable.hungry4,
        R.drawable.hungry5,
        R.drawable.hungry6,
        R.drawable.hungry7
    )

    // Tutorial açıklamaları
    private val tutorialDescriptions = listOf(
        "Bebek çok küçükken yavaş akışlı biberon ucu kullan. İlk günlerde boğulma riski daha yüksektir. Büyüdükçe, ihtiyacına göre akış hızını artırabilirsin.",
        
        "Biberon hazırlamadan önce ellerini yıka. Ilık su ve sabunla en az 20 saniye ovalayıp durula, temiz bir havluyla kurula.",
        
        "Mamayı temiz suyla ve ambalaj talimatına birebir uyarak hazırla; fazla toz gaz ve susuzluk yapar. En iyisi anne sütüdür, yoksa tek güvenli alternatif bebek mamasıdır. 6 aydan küçük bebeğe inek sütü ya da bitkisel süt verme. Kullanılan su temiz ve güvenli olsun.",
        
        "Anne sütünü gerekirse benmari ya da biberon ısıtıcısıyla ılıt; kaynatma, mikrodalga kullanma. Süt 38°C'yi geçmesin. Soğuk ya da oda sıcaklığında vermek de güvenlidir; ısıtmak şart değil.",
        
        "Beslemeden önce sütü kolunda test et. Ilık ya da serin olsun, sıcak olmasın; damlalar düzenli akmalı. Akmıyorsa uç tıkalıdır, çok akıyorsa uç bozuktur—değiştir. Süt serin olsun, bebeğin ağzını yakmasın.",
        
        "Biberon ve emziği günde bir kez sterilize et. 5 dakika kaynatabilir ya da bulaşık makinesinin steril programını kullanabilirsin. Plastiğin ısıya uygun olduğundan emin ol; bu yüzden cam biberon tercih eden çok kişi var. İlk aylarda ağıza giren her şeyi her gün sterilize etmek güvenlidir.",
        
        "Bebeği acıktığını gösterdiğinde besle. Başını çevirme, ağzını açma, emme hareketleri erken işaretlerdir; ağlamak geç kalındığını gösterir. Beslendikten kısa süre sonra ağlıyorsa açlık dışında bez, yorgunluk, sıkılma ya da üşüme/terleme gibi nedenleri kontrol et."
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        
        binding = DialogHungryTutorialBinding.inflate(LayoutInflater.from(context))
        setContentView(binding.root)

        // Full screen dialog
        window?.setLayout(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT
        )

        setupUI()
        updateStep()
    }

    private fun setupUI() {
        // Close button
        binding.btnClose.setOnClickListener {
            dismiss()
            onComplete()
        }

        // Previous button
        binding.btnPrevious.setOnClickListener {
            if (currentStep > 0) {
                currentStep--
                updateStep()
            }
        }

        // Next button
        binding.btnNext.setOnClickListener {
            if (currentStep < tutorialImages.size - 1) {
                currentStep++
                updateStep()
            } else {
                // Son adımda "Tamamla" yi tıkladı
                dismiss()
                onComplete()
            }
        }
    }

    private fun updateStep() {
        // Update step indicator
        binding.textStepIndicator.text = "Adım ${currentStep + 1}/${tutorialImages.size}"

        // Update image
        binding.imageTutorial.setImageResource(tutorialImages[currentStep])

        // Update description
        binding.textDescription.text = tutorialDescriptions[currentStep]

        // Update button visibility and text
        if (currentStep == 0) {
            binding.btnPrevious.visibility = View.INVISIBLE
        } else {
            binding.btnPrevious.visibility = View.VISIBLE
        }

        if (currentStep == tutorialImages.size - 1) {
            binding.btnNext.text = "✓ Tamamla"
        } else {
            binding.btnNext.text = "Sonraki ▶"
        }
    }
}
