# Kinematik Vuruş Algılama

## Problem Tanımı

Ofsayt kararı **topun oynanma anında** verilmelidir. Bu nedenle, vuruş anını doğru tespit etmek kritiktir.

!!! danger "Yanlış Vuruş Tespiti Sonuçları"
    - **Erken tespit:** Ofsayt çizgisi yanlış pozisyonda → Hatalı karar
    - **Geç tespit:** Çizgi çok ilerlemiş → Hatalı karar
    - **Kaçırılmış vuruş:** VAR kararı hiç verilmez

---

## Fiziksel Model

Vuruş anını tespit etmek için **Newton Kinematiği** kullanılır:

### Konum, Hız ve İvme

Topun pozisyonu $\mathbf{p}(t)$, hızı $\mathbf{v}(t)$ ve ivmesi $\mathbf{a}(t)$:

$$
\mathbf{v}(t) = \mathbf{p}(t) - \mathbf{p}(t-1)
$$

$$
\mathbf{a}(t) = \mathbf{v}(t) - \mathbf{v}(t-1) = \mathbf{p}(t) - 2\mathbf{p}(t-1) + \mathbf{p}(t-2)
$$

### Vuruş Fiziği

Bir vuruş anında:
1. **İvme artışı:** Top aniden hızlanır
2. **Yön değişimi:** Hareket vektörü değişir

```
Vuruş Öncesi        Vuruş Anı           Vuruş Sonrası
     ○                  ○                    ○
     ↓                 ╱ ← Δv büyük          ↗
     ○                ○                    ○
     ↓                                      
     ○ (yavaş)        (hızlı)           (çok hızlı)
```

---

## İvme Vektörü Analizi

### Hesaplama

```python
def calculate_acceleration(ball_history):
    """
    Son 3 frame'deki pozisyonlardan ivme hesaplar.
    
    Args:
        ball_history: [(x1,y1), (x2,y2), (x3,y3)]
        
    Returns:
        float: İvme büyüklüğü (m/frame²)
    """
    p1, p2, p3 = ball_history[-3:]
    
    # İvme vektörü: a = p3 - 2*p2 + p1
    accel_vector = np.array(p3) - 2*np.array(p2) + np.array(p1)
    
    # İvme büyüklüğü (magnitude)
    accel_magnitude = np.linalg.norm(accel_vector)
    
    return accel_magnitude
```

### Formül

$$
\|\mathbf{a}\| = \|\mathbf{p}(t) - 2\mathbf{p}(t-1) + \mathbf{p}(t-2)\|
$$

### Threshold Belirleme

```python
ACCEL_THRESHOLD = 1.5  # m/frame²
```

| Hareket | İvme (m/frame²) |
|---------|-----------------|
| Durağan top | 0 - 0.3 |
| Yavaş yuvarlanma | 0.3 - 0.8 |
| Pas | 0.8 - 1.5 |
| **Şut/Pas** | **1.5+** ← Threshold |
| Sert şut | 2.5+ |

---

## Yön Değişimi Analizi

### Neden Sadece İvme Yetmez?

Bazı vuruşlar **düşük ivmeli** olabilir (yumuşak pas). Bu durumda **yön değişimi** daha güvenilir bir sinyal olur.

### Dot Product ile Yön Benzerliği

$$
\cos\theta = \frac{\mathbf{v}_1 \cdot \mathbf{v}_2}{\|\mathbf{v}_1\| \|\mathbf{v}_2\|}
$$

| $\cos\theta$ | Anlam |
|--------------|-------|
| 1.0 | Aynı yön (düz hareket) |
| 0.0 | Dik açı (90° dönüş) |
| -1.0 | Ters yön (180° dönüş) |

```python
def calculate_direction_change(ball_history):
    """Yön değişimini hesaplar (dot product)."""
    p1, p2, p3 = ball_history[-3:]
    
    # Hız vektörleri
    v1 = np.array([p2[0]-p1[0], p2[1]-p1[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    
    # Normalize et
    v1_unit = v1 / (np.linalg.norm(v1) + 1e-6)
    v2_unit = v2 / (np.linalg.norm(v2) + 1e-6)
    
    # Dot product: 1=aynı yön, 0=dik, -1=ters yön
    direction_similarity = np.dot(v1_unit, v2_unit)
    
    return direction_similarity
```

### Threshold

```python
DIR_CHANGE_THRESHOLD = 0.3  # cos(θ) < 0.3 → ~73° üzeri dönüş
```

---

## Proximity Constraint

### Fiziksel Kural

Bir vuruş için topun **bir oyuncunun yakınında** olması gerekir:

$$
d_{\min} = \min_{p \in \text{Players}} \|\mathbf{ball} - \mathbf{p}\|
$$

```python
def calculate_min_distance(ball_coord, players):
    """Top ile en yakın oyuncu arasındaki mesafe."""
    distances = [
        np.linalg.norm(ball_coord - p['coord']) 
        for p in players
    ]
    return min(distances) if distances else float('inf')
```

### Threshold

```python
DIST_THRESHOLD = 1.8  # metre
```

| Mesafe | Anlam |
|--------|-------|
| < 1.0m | Top ayakta veya kontrolde |
| 1.0-1.8m | Vuruş menzilinde |
| > 1.8m | Oyuncudan uzak (vuruş olamaz) |

---

## Karar Mantığı

### Kombine Kriterler

Vuruş tespiti için **iki koşul birden** sağlanmalı:

$$
\text{Kick} = \underbrace{(\|\mathbf{a}\| > 1.5 \lor \cos\theta < 0.3)}_{\text{Fiziksel Sinyal}} \land \underbrace{(d_{\min} < 1.8)}_{\text{Yakınlık}}
$$

```python
def analyze(self, frame_idx, ball_coord, players) -> bool:
    # ... önceki kontroller ...
    
    # Fiziksel sinyaller
    is_physically_kick = (accel > self.accel_threshold or 
                          dir_change < 0.3)
    
    # Yakınlık kontrolü
    is_near_enough = min_dist < self.dist_threshold
    
    # Final karar
    if is_physically_kick and is_near_enough:
        self.last_kick_frame = frame_idx
        return True
    
    return False
```

### Neden OR (∨) ve AND (∧)?

- **Fiziksel sinyal (OR):** Yumuşak pas düşük ivmeli ama yön değiştirir
- **Yakınlık (AND):** Topa dokunulmadan ivme artışı olamaz (teknik hata hariç)

---

## Cooldown Mekanizması

### Problem: Çoklu Tetikleme

Tek bir vuruş, birden fazla frame'de threshold'u geçebilir:

```
Frame 100: accel=2.1 → KICK! ✓
Frame 101: accel=1.8 → KICK! ✗ (aynı vuruş!)
Frame 102: accel=1.6 → KICK! ✗ (aynı vuruş!)
```

### Çözüm: Cooldown

```python
def __init__(self, fps=25.0):
    self.cooldown_frames = int(fps * 0.8)  # 0.8 saniye
    self.last_kick_frame = -100

def analyze(self, frame_idx, ball_coord, players):
    # Cooldown kontrolü
    if frame_idx - self.last_kick_frame < self.cooldown_frames:
        self.telemetry["status"] = "COOLDOWN"
        return False
    
    # ... diğer analizler ...
    
    if is_kick:
        self.last_kick_frame = frame_idx  # Güncelle
        return True
```

### Cooldown Süresi

```
0.8 saniye × 25 FPS = 20 frame
```

Bu süre, ardışık paslar için yeterince kısa, tekrarlı tetiklemeyi önlemek için yeterince uzundur.

---

## Top Kaybı Toleransı

### Problem: Hızlı Hareket = Motion Blur

Hızlı hareket eden top, bazı frame'lerde tespit edilemeyebilir:

```
Frame 97: Ball at (34.2, 28.5) ✓
Frame 98: Ball NOT DETECTED ✗
Frame 99: Ball at (38.1, 25.2) ✓
```

### Çözüm: Toleranslı Kayıp

```python
def __init__(self):
    self.lost_ball_counter = 0
    self.max_lost_frames = 5  # 5 frame tolerans

def analyze(self, frame_idx, ball_coord, players):
    if ball_coord is None:
        self.lost_ball_counter += 1
        
        if self.lost_ball_counter > self.max_lost_frames:
            # Çok uzun süre kayıp → history'yi sıfırla
            self.ball_history.clear()
            return False
        else:
            # Kısa süreli kayıp → tolere et
            return False
    
    # Top bulundu, sayacı sıfırla
    self.lost_ball_counter = 0
```

---

## Telemetri Dashboard

### Durum Bilgileri

```python
telemetry = {
    "accel": 0.0,        # Anlık ivme
    "dir_change": 0.0,   # Yön benzerliği (-1 to 1)
    "min_dist": 99.0,    # En yakın oyuncuya mesafe
    "status": "ARMED",   # Durum string'i
    "reason": ""         # Tetiklenmeme nedeni
}
```

### Durum Kodları

| Status | Açıklama |
|--------|----------|
| `SEARCHING` | Top aranıyor (ilk frame'ler) |
| `BALL LOST` | Top tespit edilemedi |
| `BUFFERING` | History dolduruluyor (<3 frame) |
| `COOLDOWN` | Son vuruştan beri bekleme |
| `ARMED` | Sistem aktif, vuruş bekleniyor |
| `KICK!` | Vuruş algılandı! |

### Debug Amaçlı

```python
# Neden tetiklenmediğini anlamak için
reasons = []
if not is_physically_kick: 
    reasons.append("LOW_ACCEL")
if not is_near_enough: 
    reasons.append("TOO_FAR")
    
self.telemetry["reason"] = "|".join(reasons)
```

---

## Kod İmplementasyonu

```python
class KickDetector:
    def __init__(self, fps: float = 25.0):
        self.ball_history = deque(maxlen=10)
        self.dist_threshold = 1.8
        self.accel_threshold = 1.5
        self.last_kick_frame = -100
        self.cooldown_frames = int(fps * 0.8)
        self.lost_ball_counter = 0
        self.max_lost_frames = 5
        
        self.telemetry = {
            "accel": 0.0, "dir_change": 0.0, 
            "min_dist": 99.0, "status": "SEARCHING", "reason": ""
        }

    def analyze(self, frame_idx, ball_coord, players) -> bool:
        # 1. Top kayıp kontrolü
        if ball_coord is None:
            self.lost_ball_counter += 1
            if self.lost_ball_counter > self.max_lost_frames:
                self.ball_history.clear()
            return False
        self.lost_ball_counter = 0

        # 2. Cooldown kontrolü
        if frame_idx - self.last_kick_frame < self.cooldown_frames:
            self.telemetry["status"] = "COOLDOWN"
            return False

        # 3. History güncelle
        self.ball_history.append((ball_coord[0], ball_coord[1], frame_idx))
        if len(self.ball_history) < 3:
            self.telemetry["status"] = "BUFFERING"
            return False

        # 4. Kinematik hesaplamalar
        p1, p2, p3 = list(self.ball_history)[-3:]
        v1 = np.array([p2[0]-p1[0], p2[1]-p1[1]])
        v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
        
        accel = np.linalg.norm(v2 - v1)
        v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
        dir_change = np.dot(v1_u, v2_u)

        # 5. Mesafe kontrolü
        distances = [np.linalg.norm(ball_coord - p['coord']) for p in players]
        min_dist = min(distances) if distances else 99.0

        # 6. Telemetri güncelle
        self.telemetry.update({
            "accel": accel, "dir_change": dir_change,
            "min_dist": min_dist, "status": "ARMED"
        })

        # 7. Final karar
        is_physically_kick = (accel > self.accel_threshold or dir_change < 0.3)
        is_near_enough = min_dist < self.dist_threshold

        if is_physically_kick and is_near_enough:
            self.last_kick_frame = frame_idx
            self.telemetry["status"] = "KICK!"
            return True

        return False
```

---

## Sonraki Bölümler

- [Temporal Filtering](temporal.md)
- [Ofsayt Kuralları](../offside/rules.md)
