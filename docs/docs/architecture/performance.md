# Model Performans Analizi

## Genel Bakış

Bu bölüm, sistemde kullanılan YOLOv8 modellerinin eğitim sürecini ve performans metriklerini detaylı şekilde analiz eder. Profesyonel bir VAR sistemi için **model performansının sürekli izlenmesi ve doğrulanması** kritik öneme sahiptir.

!!! info "Neden Performans Analizi?"
    Bir production sisteminde model performansı sadece "çalışıyor" demekle ölçülmez:
    
    - **Precision/Recall dengesi** → Yanlış pozitif vs kaçırılan tespitler
    - **mAP metrikleri** → Genel tespit kalitesi
    - **Loss yakınsama** → Eğitim stabilitesi
    - **Model karşılaştırma** → En uygun mimari seçimi

---

## Model Karşılaştırma Özeti

Aşağıdaki görsel, farklı YOLOv8 varyantlarının (nano, small, medium) performans karşılaştırmasını göstermektedir:

![Model Karşılaştırma Özeti](../assets/summary_comparison.png)

### Analiz

| Model | Avantaj | Dezavantaj | Kullanım Senaryosu |
|-------|---------|------------|-------------------|
| **YOLOv8n** | Hızlı inference (~5ms) | Düşük accuracy | Edge/mobil cihazlar |
| **YOLOv8s** | Dengeli performans | Orta seviye | Genel kullanım |
| **YOLOv8m** | Yüksek accuracy | Daha yavaş (~12ms) | **Bu projede tercih edildi** |

!!! success "Seçim Gerekçesi"
    **YOLOv8m** tercih edildi çünkü:
    
    - Ofsayt kararı **santimetre hassasiyetinde** doğruluk gerektirir
    - RTX 3060 ile real-time performans hâlâ sağlanır (~28 FPS)
    - Precision/Recall dengesi kritik senaryolarda daha güvenilir

---

## Eğitim Metrikleri

### Precision Karşılaştırması

**Precision**, modelin "tespit ettim" dediği nesnelerin ne kadarının gerçekten doğru olduğunu ölçer:

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

![Precision Karşılaştırması](../assets/comparison_precision.png)

**Yorum:**
- Tüm modeller ~50 epoch sonra yakınsıyor
- YOLOv8m en yüksek final precision değerine ulaşıyor
- İlk 20 epoch'ta hızlı öğrenme, sonra stabilizasyon

---

### Recall Karşılaştırması

**Recall**, gerçekte var olan nesnelerin ne kadarının tespit edildiğini ölçer:

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

![Recall Karşılaştırması](../assets/comparison_recall.png)

!!! warning "Ofsayt Bağlamında Recall Kritik"
    Düşük recall → Oyuncu kaçırılır → Ofsayt çizgisi yanlış hesaplanır
    
    Bu nedenle **recall > 0.85** hedeflenmiştir.

---

### mAP@0.5 (Mean Average Precision)

mAP@0.5, IoU eşiği 0.5 olan tüm sınıfların ortalama precision değeridir:

![mAP@0.5 Karşılaştırması](../assets/comparison_mAP50.png)

**Gözlemler:**
- YOLOv8m: **~0.91** final mAP@0.5
- YOLOv8s: **~0.88** final mAP@0.5
- YOLOv8n: **~0.84** final mAP@0.5

---

### mAP@0.5:0.95 (Strict mAP)

mAP@0.5:0.95, IoU eşiği 0.5'ten 0.95'e kadar değişen ortalama precision değeridir. **Daha zorlu bir metriktir:**

![mAP@0.5:0.95 Karşılaştırması](../assets/comparison_mAP50-95.png)

$$
\text{mAP}_{0.5:0.95} = \frac{1}{10} \sum_{i=0}^{9} AP_{0.5 + 0.05i}
$$

!!! note "Profesyonel Standart"
    Production sistemlerde **mAP@0.5:0.95 > 0.70** hedeflenir. Bu projede **0.74** değerine ulaşılmıştır.

---

### Box Loss Yakınsaması

Box Loss, bounding box koordinatlarının tahmin hatasını ölçer:

![Box Loss Karşılaştırması](../assets/comparison_box_loss.png)

**Analiz:**
- Tüm modellerde smooth yakınsama → **Overfitting yok**
- Final loss değerleri düşük → İyi generalizasyon
- YOLOv8m en düşük final loss'a sahip

---

## Performans Heatmap

Tüm metriklerin model bazında karşılaştırmalı ısı haritası:

![Performans Heatmap](../assets/heatmaps_performance.png)

### Okuma Rehberi

| Renk | Anlam |
|------|-------|
| 🟢 Koyu Yeşil | En iyi performans |
| 🟡 Sarı | Orta seviye |
| 🔴 Kırmızı | Düşük performans |

---

## Sonuç ve Öneriler

### Bu Proje İçin

| Metrik | Hedef | Gerçekleşen | Durum |
|--------|-------|-------------|-------|
| Precision | > 0.85 | 0.89 | ✅ |
| Recall | > 0.85 | 0.87 | ✅ |
| mAP@0.5 | > 0.90 | 0.91 | ✅ |
| mAP@0.5:0.95 | > 0.70 | 0.74 | ✅ |
| Inference Speed | > 25 FPS | 28 FPS | ✅ |

### Production Önerileri

!!! tip "İyileştirme Fırsatları"
    
    1. **Data Augmentation Artırımı:** Mosaic + MixUp oranlarını artır
    2. **Ensemble:** Birden fazla model birleştirerek accuracy artır
    3. **TensorRT Export:** Inference hızını 2x artır
    4. **Continuous Training:** Yeni verilerle periyodik fine-tune

---

## Sonraki Bölümler

- [YOLOv8 Pipeline](../cv/yolo-pipeline.md) - Model inference detayları
- [Homografi](../geometry/homography.md) - Koordinat dönüşümü
