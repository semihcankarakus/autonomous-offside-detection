import torch
import numpy as np
import umap
from transformers import AutoProcessor, SiglipVisionModel
from sklearn.cluster import KMeans
from typing import List, Optional, Tuple
from collections import Counter, deque

class TemporalTeamVoter:
    def __init__(self, window_size: int = 15, stability_threshold: int = 12):
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        self.history = {} 
        self.stable_teams = {} 

    def unlock_id(self, track_id: int):
        """Kritik: Çakışma anında kilidi açar ve yeniden analize zorlar."""
        if track_id in self.stable_teams:
            del self.stable_teams[track_id]
            if track_id in self.history:
                self.history[track_id].clear()
            print(f"[RE-VERIFY] ID {track_id} kilidi açıldı.")

    def vote(self, track_id: int, current_prediction: int) -> Tuple[int, bool]:
        if track_id in self.stable_teams:
            return self.stable_teams[track_id], True
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.window_size)
        self.history[track_id].append(current_prediction)
        counts = Counter(self.history[track_id])
        most_common_team, count = counts.most_common(1)[0]
        if len(self.history[track_id]) == self.window_size and count >= self.stability_threshold:
            self.stable_teams[track_id] = most_common_team
            return most_common_team, True
        return most_common_team, False

class ProfessionalTeamClassifier:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.processor = AutoProcessor.from_pretrained('google/siglip-base-patch16-224')
        self.model = SiglipVisionModel.from_pretrained('google/siglip-base-patch16-224').to(self.device)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2, n_init=10)
        self.is_trained = False
        self.calibration_buffer = []

    @torch.no_grad()
    def extract_jersey_embeddings(self, crops: list) -> np.ndarray:
        if not crops: return np.array([])
        jersey_crops = []
        for c in crops:
            h, w = c.shape[:2]
            y_start, y_end = int(h * 0.20), int(h * 0.50)
            x_start, x_end = int(w * 0.25), int(w * 0.75)
            roi = c[y_start:y_end, x_start:x_end]
            jersey_crops.append(roi if roi.size > 0 else c)
        inputs = self.processor(images=jersey_crops, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()

    def calibrate_step(self, crop: np.ndarray, limit: int = 30):
        if self.is_trained: return
        self.calibration_buffer.append(crop)
        if len(self.calibration_buffer) >= limit:
            self.calibrate(self.calibration_buffer)
            self.calibration_buffer = []

    def calibrate(self, crops: list):
        print("[INFO] SigLIP Takım Kalibrasyonu Başlıyor...")
        features = self.extract_jersey_embeddings(crops)
        projections = self.reducer.fit_transform(features)
        self.cluster_model.fit(projections)
        self.is_trained = True
        print("[INFO] Kalibrasyon Tamamlandı.")

    def predict(self, crop: np.ndarray) -> int:
        if not self.is_trained: return -1
        feature = self.extract_jersey_embeddings([crop])
        projection = self.reducer.transform(feature)
        return int(self.cluster_model.predict(projection)[0])

class OffsideLogic:
    def __init__(self, fps: float = 25.0):
        self.attacking_team_id = None
        self.attack_direction = "L2R"
        self.possession_history = deque(maxlen=int(fps * 2))
        self.smoothed_line_x = None
        self.raw_line_x = None # Gecikmesiz gerçek pozisyon
        self.alpha = 0.15 # Görsel yumuşatma
        self.stability_threshold = 0.75

    def update_attacking_team(self, players: list, ball_m: Optional[np.ndarray]):
        if not players: return
        if ball_m is not None:
            distances = [(p['team'], np.linalg.norm(ball_m - p['coord'])) for p in players if p['team'] != -1]
            if distances:
                closest_team = min(distances, key=lambda x: x[1])[0]
                self.possession_history.append(closest_team)
                counts = Counter(self.possession_history)
                most_common, count = counts.most_common(1)[0]
                # Hysteresis: %75 baskınlık olmadan takımı değiştirme
                if count / len(self.possession_history) > self.stability_threshold:
                    self.attacking_team_id = most_common

        # Atak Yönü: Team Centroid
        t0_x = [p['coord'][0] for p in players if p['team'] == 0]
        t1_x = [p['coord'][0] for p in players if p['team'] == 1]
        if len(t0_x) > 3 and len(t1_x) > 3:
            c0, c1 = np.mean(t0_x), np.mean(t1_x)
            if self.attacking_team_id == 0:
                self.attack_direction = "L2R" if c0 < c1 else "R2L"
            else:
                self.attack_direction = "L2R" if c1 < c0 else "R2L"

    def calculate_offside_line(self, players: list) -> Optional[float]:
        if self.attacking_team_id is None: return None
        defenders = [p['coord'][0] for p in players if p['team'] != self.attacking_team_id]
        if len(defenders) < 2: return self.smoothed_line_x
        
        defenders.sort(reverse=(self.attack_direction == "L2R"))
        self.raw_line_x = defenders[1] # Karar anı için gerçek değer

        if self.smoothed_line_x is None:
            self.smoothed_line_x = self.raw_line_x
        else:
            self.smoothed_line_x = (self.alpha * self.raw_line_x) + (1 - self.alpha) * self.smoothed_line_x
        return self.smoothed_line_x

class KickDetector:
    def __init__(self, fps: float = 25.0):
        self.ball_history = deque(maxlen=10)
        self.dist_threshold = 1.8
        self.accel_threshold = 2.0
        self.last_kick_frame = -100
        self.cooldown_frames = int(fps * 0.8)
        self.telemetry = {"accel": 0.0, "status": "SEARCHING", "min_dist": 99.0, "reason": ""}

    def analyze(self, frame_idx: int, ball_coord: Optional[np.ndarray], players: list) -> bool:
        if ball_coord is None or frame_idx - self.last_kick_frame < self.cooldown_frames:
            return False
        self.ball_history.append((ball_coord[0], ball_coord[1]))
        if len(self.ball_history) < 3: return False
        
        p1, p2, p3 = list(self.ball_history)[-3:]
        accel = np.linalg.norm(np.array(p3) - 2*np.array(p2) + np.array(p1))
        min_dist = min([np.linalg.norm(ball_coord - p['coord']) for p in players]) if players else 99.0
        
        self.telemetry.update({"accel": accel, "min_dist": min_dist, "status": "ARMED"})
        if accel > self.accel_threshold and min_dist < self.dist_threshold:
            self.last_kick_frame = frame_idx
            self.telemetry["status"] = "KICK!"
            return True
        return False

class FinalDecisionLogic:
    def __init__(self):
        self.last_result = {}
    def process_decision(self, attackers: list, offside_line_x: float, direction: str):
        if not attackers or offside_line_x is None: return None
        attackers.sort(key=lambda p: p['coord'][0], reverse=(direction == "L2R"))
        lead = attackers[0]
        diff = offside_line_x - lead['coord'][0]
        # L2R ise diff > 0 olmalı, R2L ise tam tersi. Mantığı basitleştiriyoruz:
        is_off = (direction == "L2R" and lead['coord'][0] > offside_line_x) or \
                 (direction == "R2L" and lead['coord'][0] < offside_line_x)
        self.last_result = {"is_offside": is_off, "margin": abs(diff)}
        return self.last_result