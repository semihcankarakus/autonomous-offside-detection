# Koordinat Sistemleri

## İki Koordinat Dünyası

Bu sistemde iki farklı koordinat sistemi kullanılır ve sürekli aralarında dönüşüm yapılır:

| Sistem | Birim | Orijin | Kullanım |
|--------|-------|--------|----------|
| **Piksel** | px | Sol üst köşe (0, 0) | Görüntü işleme, UI |
| **Metrik** | metre | Sol üst köşe (0, 0) | Fiziksel hesaplamalar |

```
Piksel Koordinat Sistemi         Metrik Koordinat Sistemi
─────────────────────────        ─────────────────────────
(0,0)────────────→ U            (0,0)────────────→ X
│                                │
│     Görüntü                    │     Saha
│     1920 × 1080                │     105m × 68m
│                                │
↓                                ↓
V                                Y
```

---

## FIFA Standart Saha Ölçüleri

Tüm metrik hesaplamalar **FIFA standart ölçüleri** baz alır:

```
                        105 metre
    ←───────────────────────────────────────────────→
    
    ┌───────────────────────────────────────────────┐  ↑
    │                                               │  │
    │  ┌─────────┐                     ┌─────────┐  │  │
    │  │         │                     │         │  │  │
 16.5m │   ┌─┐   │        ○            │   ┌─┐   │  │  │
    │  │   │ │   │      r=9.15m        │   │ │   │  │ 68m
    │  │   └─┘   │                     │   └─┘   │  │  │
    │  │  5.5m   │                     │  5.5m   │  │  │
    │  └─────────┘                     └─────────┘  │  │
    │     40.3m                           40.3m     │  │
    └───────────────────────────────────────────────┘  ↓
    
    ←────────────────52.5m────────────────→
                  (orta saha)
```

### Kritik Ölçüler

| Alan | Boyut |
|------|-------|
| Toplam saha | 105m × 68m |
| Ceza sahası | 40.3m × 16.5m |
| Kale alanı | 18.3m × 5.5m |
| Penaltı noktası | 11m (kale çizgisinden) |
| Orta daire yarıçapı | 9.15m |

---

## Oyuncu Koordinat Dönüşümü

### Bounding Box → Zemin Noktası

Oyuncunun saha pozisyonu için **bounding box'ın alt orta noktası** kullanılır:

```
     ┌─────────┐
     │         │
     │  Player │
     │         │
     │         │
     └────●────┘
          ↑
     (x1+x2)/2, y2
     
     "Ayak Noktası"
```

```python
# Bounding box: [x1, y1, x2, y2]
foot_point = np.array([
    [(box[0] + box[2]) / 2,  # X: Orta
     box[3]]                  # Y: Alt kenar
])

# Metre koordinatına dönüştür
player_meter = geometry.pixel_to_pitch(foot_point)
```

!!! info "Neden Alt Orta Nokta?"
    - Homografi **zemin düzleminde** çalışır
    - Oyuncunun kafa pozisyonu perspektif bozulmasına maruz kalır
    - Ayak pozisyonu, sahadaki gerçek konumu temsil eder

---

## Top Koordinat Dönüşümü

Top için de **alt kenar** kullanılır:

```python
if len(b_res.boxes) > 0:
    bx = b_res.boxes[0].xyxy[0].cpu().numpy()
    
    ball_point = np.array([
        [(bx[0] + bx[2]) / 2,  # X merkezi
         bx[3]]                # Y alt (zemine temas)
    ])
    
    ball_meter = geometry.pixel_to_pitch(ball_point)
```

---

## Ofsayt Çizgisi Koordinatları

Ofsayt çizgisi, **sabit X değerinde dikey bir doğrudur**:

### Metrik Tanım

```
X = 68 metre (örnek)

Y = 0  ──────────────────  (üst taç çizgisi)
       │
       │    Ofsayt
       │    Çizgisi
       │
Y = 68 ──────────────────  (alt taç çizgisi)
```

### Piksel Dönüşümü

```python
def get_offside_line(self, x_meter: float):
    # Sahadaki iki nokta (dikey çizginin uçları)
    pitch_points = np.array([
        [x_meter, 0],           # Üst taç çizgisi
        [x_meter, 68]           # Alt taç çizgisi
    ])
    
    # Piksele dönüştür
    pixel_points = self.pitch_to_pixel(pitch_points)
    
    return pixel_points
```

### Perspektif Etkisi

Metrik koordinatta dikey olan çizgi, piksel koordinatında **eğik** görünür:

```
Metrik (Kuşbakışı)           Piksel (Perspektif)
─────────────────            ──────────────────
     │                           ╲
     │                            ╲
     │                             ╲
     │                              ╲
     │                               ╲
```

---

## Mesafe Hesaplamaları

### Oyuncu-Top Mesafesi

```python
def euclidean_distance(p1, p2):
    """
    İki nokta arası Öklid mesafesi (metre).
    
    Args:
        p1, p2: [x, y] metre koordinatları
    """
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
```

$$
d = \sqrt{(x_1 - x_2)^2 + (y_1 - y_2)^2}
$$

### Ofsayt Marjı

```python
# Hücumcu pozisyonu: attacker_x
# Ofsayt çizgisi: offside_line_x

margin = abs(attacker_x - offside_line_x)  # metre cinsinden
```

---

## Koordinat Validasyonu

### Saha Sınırları Kontrolü

```python
def is_on_pitch(coord: np.ndarray) -> bool:
    """Koordinatın saha içinde olup olmadığını kontrol eder."""
    x, y = coord
    return (0 <= x <= 105) and (0 <= y <= 68)
```

### Aykırı Değer Tespiti

```python
def validate_player_position(coord: np.ndarray) -> bool:
    """
    Fiziksel olarak imkansız pozisyonları filtreler.
    """
    x, y = coord
    
    # Saha dışı
    if x < -5 or x > 110 or y < -5 or y > 73:
        return False
    
    # Kale arkası (sadece kaleci olabilir)
    if x < 0 or x > 105:
        return False
    
    return True
```

---

## Atak Yönü ve Koordinat Sistemi

### L2R (Soldan Sağa)

```
Hücum Yönü →

Savunma          Ofsayt Çizgisi          Hücum
   ○○                  │                   ○
   ○○                  │                  ○○
                       │
   
X: 0 ────────────────────────────────────→ X: 105
```

- Savunmacılar: Düşük X değerleri
- Ofsayt: Hücumcu X > Çizgi X → **OFFSIDE**

### R2L (Sağdan Sola)

```
                           ← Hücum Yönü

Hücum          Ofsayt Çizgisi          Savunma
  ○                    │                   ○○
 ○○                    │                   ○○
                       │

X: 0 ────────────────────────────────────→ X: 105
```

- Savunmacılar: Yüksek X değerleri
- Ofsayt: Hücumcu X < Çizgi X → **OFFSIDE**

---

## Koordinat Dönüşüm Performansı

!!! tip "Vektörizasyon"
    NumPy'nin matris çarpımı kullanıldığı için dönüşümler **vektörize** edilmiştir. 100 oyuncu için tek bir matris çarpımı yeterlidir.

---

## Sonraki Bölümler

- [SigLIP Embeddings](../classification/siglip.md)
- [UMAP & K-Means](../classification/clustering.md)
