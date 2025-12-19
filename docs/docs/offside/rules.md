# FIFA Ofsayt Kuralları

## Resmi Tanım

!!! quote "FIFA Laws of the Game - Law 11"
    Bir oyuncu, **topun ve sondan ikinci rakibin** daha gerisinde olduğunda ofsayt pozisyonundadır.
    
    Ofsayt pozisyonunda olmak tek başına suç değildir. Oyuncu ancak **topun oynanma anında** ofsayt pozisyonundaysa ve aktif olarak oyuna katılırsa ofsayttır.

---

## Kritik Kavramlar

### 1. Ofsayt Pozisyonu

Bir oyuncu şu durumlarda ofsayt **pozisyonundadır**:

- Rakip yarı sahada
- Toptan daha ileride (kale çizgisine yakın)
- Sondan ikinci rakipten daha ileride

```
         Kale Çizgisi
              │
    Kaleci → ○│
              │
 Savunmacı → ○│─────────── Ofsayt Çizgisi
              │
              │     ○ ← Hücumcu (OFFSIDE!)
              │
              │        ○ ← Top
              │
              │
    ────────────────────── Orta Saha
```

### 2. "Sondan İkinci Rakip"

Genellikle:
- **Kaleci** = Son rakip
- **Son savunmacı** = Sondan ikinci rakip

!!! warning "Kaleci İstisnası"
    Kaleci, savunmacılardan daha ilerideyse (örn: hücuma çıkmış), ofsayt çizgisi **iki savunmacıya** göre belirlenir.

### 3. "Topun Oynanma Anı"

Ofsayt, **pas verildiği an** değerlendirilir:
- Pas alan oyuncunun pozisyonu
- O andaki ofsayt çizgisi

```
t=0 (Pas anı)        t=0.5s (Top ulaştığında)
    │                      │
○ ← Oyuncu (OFFSIDE)      ○│ ← Artık çizginin gerisinde
    │                      │
────┼─── Offside Line      │
    │                      │
   ⚽ ← Top               ⚽ ← Top burada
```

---

## Ofsayt Sayılmayan Durumlar

### 1. Kendi Yarı Sahasında

```
    ─────────────────── Orta Saha
              │
    Hücumcu → ○ (Kendi yarısında = OFFSIDE YOK)
              │
```

### 2. Top ve Sondan İkinci Rakiple Aynı Hizada

```
Savunmacı → ○─────────── Ofsayt Çizgisi
            │
Hücumcu →  ○ (Aynı hizda = OFFSIDE YOK)
            │
```

### 3. Taç, Korner, Aut

Bu durumlarda ofsayt kuralı **uygulanmaz**.

---

## Vücut Parçaları

!!! info "Hangi Vücut Parçaları Sayılır?"
    Sadece **gol atılabilecek** vücut parçaları:
    
    - Kafa
    - Gövde
    - Bacaklar
    
    **Sayılmayan:** Kollar, eller

### Pratik Uygulama

```
       ┌─┐ Kafa
       ├─┤
      ┌┴─┴┐ 
      │   │ Gövde ← Bu noktalar değerlendirilir
      └┬─┬┘
       │ │ Bacaklar ← Bu noktalar değerlendirilir
       │ │
      ─┴─┴─
```

Sistemde **ayak pozisyonu** (bounding box alt kenarı) kullanılır.

---

## Bu Sistemin Yaklaşımı

### Basitleştirmeler

| FIFA Kuralı | Sistem İmplementasyonu |
|-------------|------------------------|
| Tüm vücut parçaları | Sadece ayak pozisyonu |
| Aktif katılım analizi | Tüm hücumcular değerlendirilir |
| Taç/korner istisnası | Uygulanmıyor (manuel kontrol) |

### Avantajlar

1. **Hesaplama basitliği:** Tek nokta karşılaştırması
2. **Tutarlılık:** Her durumda aynı metot
3. **Görselleştirme kolaylığı:** Tek çizgi

### Dezavantajlar

1. Gövde ofsaytta, ayak çizgide → Kaçırılır
2. Taç atışında ofsayt verebilir (yanlış pozitif)

---

## Saha Koordinat Sistemi

### X Ekseni = Ofsayt Ekseni

FIFA sahasında X koordinatı ofsayt belirler:

```
X = 0                                    X = 105
   ├────────────────────────────────────────┤
   │                                        │
  Kale                                    Kale
(Sol)                                   (Sağ)
```

### Atak Yönüne Göre Ofsayt

**L2R (Soldan Sağa Atak):**
```
Hücumcu X > Ofsayt Çizgisi X → OFFSIDE
```

**R2L (Sağdan Sola Atak):**
```
Hücumcu X < Ofsayt Çizgisi X → OFFSIDE
```

---

## Ofsayt Çizgisi Hesaplama

### Adımlar

1. **Savunmacıları filtrele:** Hücum eden takım olmayan oyuncular
2. **X koordinatına göre sırala:** Atak yönüne bağlı
3. **İkinci oyuncuyu al:** Ofsayt çizgisi

```python
def calculate_offside_line(self, players):
    # Savunmacıları filtrele
    defenders = [
        p['coord'][0] for p in players 
        if p['team'] != self.attacking_team_id
    ]
    
    if len(defenders) < 2:
        return None
    
    # Atak yönüne göre sırala
    if self.attack_direction == "L2R":
        defenders.sort(reverse=True)  # Büyükten küçüğe
    else:
        defenders.sort()  # Küçükten büyüğe
    
    # İkinci savunmacı = ofsayt çizgisi
    offside_line_x = defenders[1]
    
    return offside_line_x
```

### Örnek: L2R Atak

```
Savunmacılar X: [85, 72, 68, 45]
Sıralı (desc): [85, 72, 68, 45]
                 ↑   ↑
              Kaleci  Ofsayt Çizgisi (X=72)
```

---

## Edge Cases

### 1. Yetersiz Savunmacı

```python
if len(defenders) < 2:
    return None  # Karar verilemez
```

### 2. Kaleci Hücumda

Kaleci top kontrolünde veya ileride → Ofsayt çizgisi 2 saha oyuncusuna göre:

```
Kaleci X = 60 (hücumda)
Savunmacılar: [55, 50, 45]
Sıralı: [55, 50, 45]
         ↑   ↑
      1st  2nd → Ofsayt Çizgisi = 50
```

### 3. Hücumcu Kendi Yarısında

X < 52.5 (orta saha) → Ofsayt değerlendirmesi yapılmaz.

---

## Sonraki Bölümler

- [Karar Algoritması](algorithm.md) - Implementasyon detayları
- [Sonuçlar](../results.md) - Test ve performans
