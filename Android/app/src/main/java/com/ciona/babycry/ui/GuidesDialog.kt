package com.ciona.babycry.ui

import android.app.Dialog
import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.Window
import android.view.WindowManager
import com.ciona.babycry.databinding.DialogGuidesBinding

/**
 * Rehberler Diyaloğu
 * 3 farklı rehbere erişim sağlar
 */
class GuidesDialog(
    context: Context
) : Dialog(context, android.R.style.Theme_Black_NoTitleBar_Fullscreen) {

    private lateinit var binding: DialogGuidesBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        
        binding = DialogGuidesBinding.inflate(LayoutInflater.from(context))
        setContentView(binding.root)

        // Full screen dialog
        window?.setLayout(
            WindowManager.LayoutParams.MATCH_PARENT,
            WindowManager.LayoutParams.MATCH_PARENT
        )

        setupUI()
    }

    private fun setupUI() {
        // Close button
        binding.btnClose.setOnClickListener {
            dismiss()
        }

        // Besleme Rehberi
        binding.cardHungry.setOnClickListener {
            HungryTutorialDialog(context) {
                // Tutorial tamamlandığında bir şey yapma, guides dialog açık kalsın
            }.show()
        }

        // Gaz Çıkarma Rehberi
        binding.cardBurping.setOnClickListener {
            BurpingTutorialDialog(context) {
                // Tutorial tamamlandığında bir şey yapma
            }.show()
        }

        // Uyku Rehberi
        binding.cardTired.setOnClickListener {
            TiredTutorialDialog(context) {
                // Tutorial tamamlandığında bir şey yapma
            }.show()
        }
    }
}
