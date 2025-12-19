# Sistem Mimarisi: Genel Bakış

## Mimari Felsefe

Bu sistem, **Clean Architecture** ve **Separation of Concerns** prensipleri üzerine inşa edilmiştir. Her modül tek bir sorumluluğa sahiptir ve modüller arasındaki bağımlılıklar minimize edilmiştir.

!!! info "Tasarım Kararı"
    Monolitik bir yapı yerine **katmanlı mimari** tercih edilmiştir. Bu yaklaşım:
    
    - Unit test yazımını kolaylaştırır
    - Modüllerin bağımsız geliştirilmesine olanak tanır
    - Debugging ve profiling süreçlerini basitleştirir

---

## Katmanlı Mimari

```mermaid
flowchart TB
    subgraph Presentation["🖥️ Presentation Layer"]
        UI[OpenCV Visualization]
        Dashboard[Telemetry Dashboard]
    end
    
    subgraph Orchestration["🎯 Orchestration Layer"]
        VAR[AutonomousVAR<br/>main.py]
    end
    
    subgraph Logic["⚙️ Business Logic Layer"]
        TC[Team Classifier]
        OL[Offside Logic]
        KD[Kick Detector]
        FD[Final Decision]
    end
    
    subgraph Core["🔧 Core Services Layer"]
        IE[Inference Engine]
        GE[Geometry Engine]
    end
    
    subgraph Infrastructure["📦 Infrastructure Layer"]
        YOLO[YOLOv8 Models]
        SigLIP[SigLIP Model]
        CUDA[CUDA Runtime]
    end
    
    UI --> VAR
    Dashboard --> VAR
    VAR --> TC & OL & KD & FD
    TC --> IE & GE
    OL --> GE
    KD --> GE
    FD --> OL
    IE --> YOLO & SigLIP
    YOLO & SigLIP --> CUDA
```

---

## Veri Akışı (Data Flow)

Sistem, her frame için aşağıdaki pipeline'ı işletir:

```mermaid
sequenceDiagram
    autonumber
    participant V as Video Stream
    participant IE as Inference Engine
    participant GE as Geometry Engine
    participant TC as Team Classifier
    participant KD as Kick Detector
    participant OL as Offside Logic
    participant FD as Final Decision
    participant UI as Visualization

    V->>IE: Frame (BGR Image)
    
    par Parallel Detection
        IE->>IE: predict_field()
        IE->>IE: predict_players()
        IE->>IE: predict_ball()
    end
    
    IE->>GE: Field Keypoints
    GE->>GE: solve_homography()
    
    IE->>GE: Player Bounding Boxes
    GE->>TC: Player Coordinates (meters)
    
    TC->>TC: extract_embeddings()
    TC->>TC: cluster_teams()
    
    TC->>OL: Players with Team IDs
    IE->>KD: Ball Coordinates
    
    KD->>KD: analyze_kinematics()
    
    alt Kick Detected
        KD->>FD: Trigger Decision
        OL->>FD: Offside Line
        FD->>UI: VAR Decision
    else No Kick
        OL->>UI: Update Visualization
    end
```

---

## Ana Bileşenler

### 1. AutonomousVAR (Orchestrator)

**Dosya:** `main.py`

Tüm alt sistemleri koordine eden ana sınıf:

```python
class AutonomousVAR:
    def __init__(self, video_path, output_path):
        # Engine Initialization
        self.inference = InferenceEngine({...})
        self.geometry = GeometryEngine()
        self.team_classifier = ProfessionalTeamClassifier()
        self.team_voter = TemporalTeamVoter()
        self.kick_detector = KickDetector(fps=self.fps)
        self.offside_logic = OffsideLogic(fps=self.fps)
        self.final_logic = FinalDecisionLogic()
```

**Sorumluluklar:**

| Görev | Açıklama |
|-------|----------|
| Video I/O | Frame okuma ve video yazma |
| Engine Koordinasyonu | Alt sistemlerin sıralı çağrımı |
| State Management | Freeze frame, calibration durumu |
| Visualization | UI rendering ve dashboard |

---

### 2. InferenceEngine

**Dosya:** `inference.py`

Tüm deep learning model çıkarımlarını yönetir:

```python
class InferenceEngine:
    def __init__(self, model_paths: Dict[str, str]):
        self.field_model = YOLO(model_paths['field'], task='pose')
        self.player_model = YOLO(model_paths['player'], task='detect')
        self.ball_model = YOLO(model_paths['ball'], task='detect')
```

!!! warning "Performans Notu"
    Modeller **half precision (FP16)** modunda çalıştırılır. Bu, VRAM kullanımını ~%50 azaltır ve inference süresini kısaltır.

---

### 3. GeometryEngine

**Dosya:** `geometry.py`

Koordinat dönüşümlerini yönetir:

```python
class GeometryEngine:
    def solve_from_model(self, keypoints, confidences) -> bool:
        """RANSAC ile robust homografi hesaplar"""
        
    def pixel_to_pitch(self, points) -> np.ndarray:
        """Piksel → Metre dönüşümü"""
        
    def pitch_to_pixel(self, points) -> np.ndarray:
        """Metre → Piksel dönüşümü"""
```

---

### 4. Logic Layer

**Dosya:** `logic.py`

İş mantığını içeren sınıflar:

| Sınıf | Sorumluluk |
|-------|------------|
| `ProfessionalTeamClassifier` | SigLIP + UMAP + K-Means takım sınıflandırma |
| `TemporalTeamVoter` | Gürültülü tahminleri stabilize etme |
| `OffsideLogic` | Atak yönü ve ofsayt çizgisi hesaplama |
| `KickDetector` | Kinematik vuruş algılama |
| `FinalDecisionLogic` | VAR kararı üretme |

---

## Durum Makinesi (State Machine)

Sistem, iki ana durumda çalışır:

```mermaid
stateDiagram-v2
    [*] --> Calibrating
    
    Calibrating --> Active: buffer_size >= 30
    Calibrating --> Calibrating: buffer_size < 30
    
    Active --> Processing: frame_received
    Processing --> KickDetected: kick == true
    Processing --> Active: kick == false
    
    KickDetected --> Frozen: freeze_frames > 0
    Frozen --> Active: freeze_frames == 0
    
    Active --> [*]: video_end
```

### Durumlar

| Durum | Açıklama |
|-------|----------|
| **Calibrating** | Takım sınıflandırıcı eğitiliyor (ilk 30 frame) |
| **Active** | Normal işleme modu |
| **Processing** | Frame analizi devam ediyor |
| **KickDetected** | Vuruş algılandı, VAR kararı bekleniyor |
| **Frozen** | Karar ekranı gösteriliyor (3 saniye) |

---

## Bağımlılık Grafiği

```mermaid
graph TD
    A[main.py] --> B[inference.py]
    A --> C[geometry.py]
    A --> D[logic.py]
    A --> E[kick_detector.py]
    
    D --> F[transformers<br/>SigLIP]
    D --> G[sklearn<br/>KMeans]
    D --> H[umap-learn]
    
    B --> I[ultralytics<br/>YOLOv8]
    
    C --> J[opencv-python]
    C --> K[numpy]
    
    subgraph External
        F
        G
        H
        I
        J
        K
    end
```

---

## Konfigürasyon Parametreleri

```python
# Inference Thresholds
FIELD_CONF = 0.5      # Saha keypoint güven eşiği
PLAYER_CONF = 0.4     # Oyuncu tespit güven eşiği
BALL_CONF = 0.3       # Top tespit güven eşiği

# Geometry
MIN_KEYPOINTS = 6     # Homografi için minimum nokta
RANSAC_THRESHOLD = 3.0 # RANSAC outlier eşiği

# Team Classification
CALIBRATION_BUFFER = 30  # Eğitim için gerekli frame sayısı
UMAP_COMPONENTS = 3       # UMAP çıktı boyutu
KMEANS_CLUSTERS = 2       # Takım sayısı

# Kick Detection
ACCEL_THRESHOLD = 1.5    # İvme eşiği (m/frame²)
DIST_THRESHOLD = 1.8     # Oyuncu-top mesafe eşiği (m)
COOLDOWN_FRAMES = 20     # Vuruşlar arası minimum frame

# Offside Logic
SMOOTHING_ALPHA = 0.15   # EMA yumuşatma faktörü
STABILITY_THRESHOLD = 0.75  # Takım değişimi için gerekli oran
```

---

## Sonraki Bölümler

- [Modül Yapısı](modules.md) - Her modülün detaylı API dokümantasyonu
- [Computer Vision Pipeline](../cv/yolo-pipeline.md) - YOLOv8 model detayları
