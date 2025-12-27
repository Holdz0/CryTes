package com.ciona.babycry.ui

import android.app.Dialog
import android.content.Context
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.Window
import android.view.WindowManager
import com.ciona.babycry.R
import com.ciona.babycry.databinding.DialogTiredTutorialBinding

/**
 * Yorgunluk/Uyku Tutorial Diyaloğu
 * 10 adımlık görsel rehber
 */
class TiredTutorialDialog(
    context: Context,
    private val onComplete: () -> Unit
) : Dialog(context, android.R.style.Theme_Black_NoTitleBar_Fullscreen) {

    private lateinit var binding: DialogTiredTutorialBinding
    private var currentStep = 0

    // Tutorial görselleri (drawable resource IDs)
    private val tutorialImages = listOf(
        R.drawable.tired1,
        R.drawable.tired2,
        R.drawable.tired3,
        R.drawable.tired4,
        R.drawable.tired5,
        R.drawable.tired6,
        R.drawable.tired7,
        R.drawable.tired8,
        R.drawable.tired9,
        R.drawable.tired10
    )

    // Tutorial açıklamaları
    private val tutorialDescriptions = listOf(
        "Mışıl mışıl uyuması ve kendini güvende hissetmesi için onu nazikçe kundağa sar. Kollarını yanlarına alıp sarmaladığında irkilip uyanmaz, huzurla dolar. Dönmeye başlayana kadar onu böyle sarıp sarmalayabilirsin.",
        
        "Bebeğinin gevşeyip uykuya hazırlanması için nazikçe masaj yap. Eline biraz yağ alıp 10-15 dakika boyunca yavaş, uzun ve orta baskılı hareketlerle; ayak tabanlarından bacaklarına, omuzlarından göğsüne kadar tüm vücudunu ov. Son olarak alnı, burnu ve başını nazikçe okşayarak işlemi tamamla.",
        
        "Bebeğini uykuya geçişte rahatlatmak için emzik verebilirsin. Emme refleksi onları sakinleştirir; beslenme bittikten sonra emzik sunmak işe yarar. Emzik kullanımı güvenlidir, hatta SIDS riskini azaltabileceğini gösteren çalışmalar var.\nKordonlu veya klipsli emzikleri beşiğe koyma; boğulma riski oluşturur.\nEmziriyorsan, meme karmaşasını önlemek için emzik vermeden önce emzirmenin oturmasını bekle—genelde doğumdan 3–4 hafta sonra.",
        
        "Sesin bebeği sakinleştirebilir. Alçak, yavaş ve tekdüze konuş. Masal işe yaramazsa; kucakta gezdir, nazikçe salla, ninni söyle ya da sakin müzik aç.",
        
        "Işıkları kısmak bebeğin beynine \"uyku vakti\" sinyali verir. Odayı gece boyunca karanlık tut; yapay ışık melatonini baskılar ve uykuyu bozar.",
        
        "Bebeği sırtüstü yatır, beşiği boş bırak. Kıpırdanırsa hemen alma; birkaç dakika kendi kendine sakinleşmesine izin ver. Battaniye, yastık, oyuncak koyma—sadece çarşaf yeter. Bu hem bağımsız uyumayı öğretir hem de SIDS riskini azaltır.",
        
        "Diş çıkarma uykuyu zorlaştırabilir. Rutini bozma; bol sarıl, diş etlerine soğuk bez uygula, gerekirse daha sık emzir. Bu dönem geçicidir. Şiddetli ağrı varsa doktora danış; uygun görürse bebekler için ağrı kesici önerebilir.",
        
        "Gece ağlıyorsa aç ya da altı kirli olabilir; mutlaka kontrol et. Yatmadan önce beslemek ve altını değiştirmek iyidir ama bebekler yine de uyanabilir. Biberona mama/gevrek eklemek önerilmez; tok tutmaz, gaz yapıp huzursuzluğu artırır.",
        
        "Gece mızmızlanınca hemen atlama; birkaç dakika bekle. Devam ederse loşta kontrol et, aç ve rahatsa uykulu halde tekrar yatır. Çok uyarmadan sakin bir şekilde sırtını okşa; bezi çok ıslak değilse sabaha kadar bekleyebilirsin.",
        
        "Uyku öncesi basit ve sakin bir rutin oluştur. Işıkları kıs, hafif sallayarak ninni söyle; büyüdükçe kısa masal, kundak, masaj ya da yumuşak müzik ekleyebilirsin. En önemlisi: her gece aynı şekilde yap, bebek uykuya hazırlanmayı öğrensin."
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        requestWindowFeature(Window.FEATURE_NO_TITLE)
        
        binding = DialogTiredTutorialBinding.inflate(LayoutInflater.from(context))
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
