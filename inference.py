import torch
from ultralytics import YOLO
from typing import Dict, Any, Optional

class InferenceEngine:
    def __init__(self, model_paths: Dict[str, str]):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[INFO] InferenceEngine: İşlemler {self.device.upper()} üzerinde yürütülecek.")
        
        # Modelleri 'None' veya boş değilse yükle
        self.field_model = self._load_model(model_paths.get('field'), task='pose')
        self.player_model = self._load_model(model_paths.get('player'), task='detect')
        self.ball_model = self._load_model(model_paths.get('ball'), task='detect')

    def _load_model(self, path: Optional[str], task: str) -> Optional[YOLO]:
        """Profesyonel Model Yükleyici: Yol geçerli değilse sessizce atlar."""
        if not path or path == '':
            return None
        
        try:
            model = YOLO(path, task=task)
            model.to(self.device)
            return model
        except Exception as e:
            print(f"[ERROR] Model yüklenemedi ({path}): {e}")
            return None

    def predict_field(self, frame) -> Any:
        if self.field_model is None: return None
        return self.field_model.predict(frame, verbose=False, device=self.device, conf=0.5, half=True)[0]

    def predict_players(self, frame, conf: float = 0.4) -> Any:
        if self.player_model is None: return None
        # Takip modunda çalıştır - persist=True ID korunumu sağlar
        return self.player_model.track(frame, persist=True, verbose=False, device=self.device, conf=conf, half=True)[0]

    def predict_ball(self, frame) -> Any:
        """Kritik Eksiklik: Top tespiti için sarmalayıcı metot."""
        if self.ball_model is None: return None
        # Top çok küçük olduğu için genellikle 'track' yerine yüksek 'conf' ile 'predict' tercih edilir
        return self.ball_model.predict(frame, verbose=False, device=self.device, conf=0.3, half=True)[0]