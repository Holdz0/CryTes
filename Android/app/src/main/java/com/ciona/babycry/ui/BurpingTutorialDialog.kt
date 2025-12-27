package com.ciona.babycry.ui

import android.app.Dialog
import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.Window
import android.view.WindowManager
import com.ciona.babycry.R
import com.ciona.babycry.databinding.DialogBurpingTutorialBinding

/**
 * Gaz/Geğirme Tutorial Diyaloğu
 * 4 adımlık görsel rehber
 */
class BurpingTutorialDialog(
    context: Context,
    private val onComplete: () -> Unit
) : Dialog(context, android.R.style.Theme_Black_NoTitleBar_Fullscreen) {

    private lateinit var binding: DialogBurpingTutorialBinding
    private var currentStep = 0

    // Tutorial görselleri (drawable resource IDs)
    private val tutorialImages = listOf(
        R.drawable.burping1,
        R.drawable.burping2,
        R.drawable.burping3,
        R.drawable.burping4
    )

    // Tutorial açıklamaları
    private val tutorialDescriptions = listOf(
        "Bebeği, karnı omzuna denk gelecek şekilde yasla; bu sırada başını ve boynunu mutlaka destekle. Kusma ihtimaline karşı (ki bu gayet normaldir) omzuna bir bez koy.",
        
        "Kürek kemiklerinin arasına, kolunu değil sadece bileğini kullanarak nazikçe vur. İstersen bunun yerine sırtını dairesel hareketlerle de ovabilirsin.",
        
        "Gaz sesini duyunca dur. Bu ses normal bir geğirme olabileceği gibi; hapşırık, hırıltı veya kısa bir \"hık\" sesi de olabilir.",
        
        "Gazını çıkarır çıkarmaz onu hemen önüne al, yüzüne kocaman ve sevgiyle gülümse. Gözlerinin içine bakıp yanında olduğunu hissettir ve minik bir öpücük kondur."
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        
        binding = DialogBurpingTutorialBinding.inflate(LayoutInflater.from(context))
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
