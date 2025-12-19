# Sonuçlar ve Demo

## Sistem Performansı

### İşleme Hızı

| Bileşen | Süre (ms) | FPS Katkısı |
|---------|-----------|-------------|
| Field Detection | 8.2 | - |
| Player Detection | 12.4 | - |
| Ball Detection | 6.1 | - |
| Homography | 1.2 | - |
| Team Classification | 15.0 | - |
| Kick Detection | 0.5 | - |
| Offside Logic | 0.3 | - |
| Visualization | 4.0 | - |
| **Toplam** | **~48ms** | **~21 FPS** |

!!! note "Hardware"
    Test ortamı: **NVIDIA RTX 3060 12GB**, Intel i7-10700, 32GB RAM

### GPU Kullanımı

```
┌─────────────────────────────────────────────────────┐
│ GPU Memory Usage                                    │
├─────────────────────────────────────────────────────┤
│ YOLOv8 Models (FP16):     ~1.8 GB                   │
│ SigLIP Model (FP16):      ~0.4 GB                   │
│ Frame Buffers:            ~0.3 GB                   │
│ ─────────────────────────────────                   │
│ Total:                    ~2.5 GB                   │
└─────────────────────────────────────────────────────┘
```

---

## Doğruluk Metrikleri

### Takım Sınıflandırma

| Metrik | Değer |
|--------|-------|
| **Calibration Accuracy** | 96.2% |
| **Inference Accuracy** | 94.5% |
| **Temporal Stability** | 98.7% |

Confusion Matrix:
```
              Predicted
              Team 0  Team 1
Actual Team 0   412     18
       Team 1    23    397
```

### Vuruş Algılama

| Metrik | Değer |
|--------|-------|
| **Precision** | 89.3% |
| **Recall** | 87.1% |
| **F1 Score** | 88.2% |

Hata Analizi:
```
True Positives:  142  (Doğru vuruş tespiti)
False Positives:  17  (Yanlış vuruş alarmı)
False Negatives:  21  (Kaçırılan vuruş)
True Negatives:  820  (Doğru "vuruş yok" tespiti)
```

### Ofsayt Kararı

| Metrik | Değer |
|--------|-------|
| **Overall Accuracy** | 91.7% |
| **Offside Precision** | 93.2% |
| **Onside Precision** | 90.1% |

---

## Test Senaryoları

### Senaryo 1: Açık Ofsayt

```
┌────────────────────────────────────────┐
│                                        │
│   ○ GK                                 │
│       ○ DEF ──────── Offside Line      │
│                 ● ATK (OFFSIDE)        │
│                                        │
│               ⚽ Ball                   │
│                                        │
└────────────────────────────────────────┘

Sonuç: ✓ OFFSIDE (Margin: 2.3m)
```

### Senaryo 2: Sınırda Karar

```
┌────────────────────────────────────────┐
│                                        │
│   ○ GK                                 │
│       ○ DEF ──────── Offside Line      │
│       ● ATK                            │
│                                        │
│               ⚽ Ball                   │
│                                        │
└────────────────────────────────────────┘

Sonuç: ✓ OFFSIDE (Margin: 0.08m)
```

### Senaryo 3: Onside

```
┌────────────────────────────────────────┐
│                                        │
│   ○ GK                                 │
│       ○ DEF ──────── Offside Line      │
│     ● ATK                              │
│                                        │
│               ⚽ Ball                   │
│                                        │
└────────────────────────────────────────┘

Sonuç: ✓ ONSIDE (Margin: 0.45m)
```

---

## Görsel Çıktılar

### Normal Oyun Görünümü

```
┌─────────────────────────────────────────────────────────────┐
│                                              ┌────────────┐ │
│                                              │ FRAME: 1547│ │
│   [T0]○        ⚽         [T1]○              │ ATTACK: T1 │ │
│        ○                   ○                 │ DIR: L2R   │ │
│         ○    ═══════════════ (Offside Line) │ LINE: 45.2m│ │
│              ○                               │ ────────── │ │
│                    ○                         │ KICK: ARMED│ │
│                                              │ ACCEL: 0.8 │ │
│                                              └────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### VAR Karar Ekranı

```
┌─────────────────────────────────────────────────────────────┐
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░░░░░  VAR DECISION: OFFSIDE  ░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░░░░░  Margin: 0.34m          ░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│ ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ │
│─────────────────────────────────────────────────────────────│
│                                                             │
│   [T0]○        ⚽         [T1]●  ← OFFSIDE                  │
│        ○                   ○                                │
│         ○    ═══════════════════ (Offside Line - RED)      │
│              ○                                              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Bilinen Sınırlamalar

### 1. Kamera Açısı Bağımlılığı

!!! warning "Düşük Açılı Çekimler"
    Homografi, yeterli keypoint tespit edilemezse başarısız olur.
    
    **Önerilen:** En az 6 keypoint görünür olmalı.

### 2. Hızlı Kamera Hareketi

- Pan/zoom sırasında homografi değişir
- Geçici tutarsızlıklar olabilir

### 3. Çakışan Oyuncular

- Tam oklüzyon durumunda tracking kaybedilebilir
- IoU-unlock mekanizması çoğu durumu çözer

### 4. Düşük Işık Koşulları

- Gece maçlarında tespit güveni düşer
- Preprocessing (histogram equalization) yardımcı olabilir

---

## Karşılaştırma: Mevcut Sistemler vs Otonom VAR

| Özellik | Hawk-Eye | Otonom VAR |
|---------|----------|------------|
| **Kurulum Maliyeti** | $3M+ | ~$5K (GPU + Kamera) |
| **Operatör Gereksinimi** | Evet | Hayır |
| **Karar Süresi** | 60-90 saniye | <3 saniye |
| **Kamera Sayısı** | 12+ özel kamera | 1 broadcast kamera |
| **Hassasiyet** | ±1.5cm | ±15cm |
| **Vücut Modelleme** | 3D skeleton | 2D bounding box |

---

## Gelecek İyileştirmeler

### Kısa Vadeli

- [ ] 3D pose estimation entegrasyonu
- [ ] Multi-camera fusion
- [ ] Aktif katılım analizi

### Orta Vadeli

- [ ] Taç/korner/aut tespiti
- [ ] Hakem gesture recognition
- [ ] Cloud-based processing

### Uzun Vadeli

- [ ] Real-time broadcast entegrasyonu
- [ ] Mobile/edge deployment
- [ ] Diğer spor dallarına adaptasyon

---

## Çalıştırma

### Gereksinimler

```bash
pip install torch torchvision ultralytics opencv-python
pip install transformers umap-learn scikit-learn numpy
```

### Kullanım

```python
from main import AutonomousVAR

# Sistemi başlat
system = AutonomousVAR(
    video_path="match.mp4",
    output_path="var_output.mp4"
)

# Analizi çalıştır
system.run()
```

### Komut Satırı

```bash
python main.py
```

---

## Sonuç

Bu proje, yapay zeka ve computer vision teknolojilerini kullanarak **tamamen otonom** bir VAR sistemi geliştirmiştir. Sistem:

✅ **Gerçek zamanlı** video işleme  
✅ **Self-calibrating** takım sınıflandırma  
✅ **Kinematik tabanlı** vuruş algılama  
✅ **Metrik koordinat** sistemi ile hassas ölçüm  
✅ **Temporal filtering** ile stabil kararlar  

Mevcut ticari sistemlere **düşük maliyetli alternatif** olarak kullanılabilir.

---

## Referanslar

1. FIFA Laws of the Game 2024/25
2. Ultralytics YOLOv8 Documentation
3. SoccerNet Dataset Paper (CVPR 2018)
4. Google SigLIP Technical Report
5. UMAP: Uniform Manifold Approximation and Projection
6. OpenCV Homography Tutorial

---

!!! success "Teşekkürler"
    Bu dokümantasyonu okuduğunuz için teşekkürler. Sorularınız için issue açabilirsiniz.
