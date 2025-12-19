from collections import deque
import numpy as np
from typing import List, Optional


class KickDetector:
    def __init__(self, fps: float = 25.0):
        self.ball_history = deque(maxlen=10)
        self.dist_threshold = 1.8 
        self.accel_threshold = 1.5  # Yumuşak pasları tespit için düşürüldü
        self.last_kick_frame = -100
        self.cooldown_frames = int(fps * 0.8)
        
        # Top kaybolma toleransı (hızlı hareketler için)
        self.lost_ball_counter = 0
        self.max_lost_frames = 5  # 5 frame boyunca kaybolmayı tolere et
        
        # DEBUG İÇİN TELEMETRİ VERİLERİ
        self.telemetry = {
            "accel": 0.0,
            "dir_change": 0.0,
            "min_dist": 99.0,
            "status": "SEARCHING", # SEARCHING, COOLDOWN, KICK!
            "reason": "" # Neden tetiklenmediğini anlamak için
        }

    def analyze(self, frame_idx: int, ball_coord: Optional[np.ndarray], players: list) -> bool:
        if ball_coord is None:
            self.lost_ball_counter += 1
            # Sadece max_lost_frames'den fazla kaybolursa history'yi temizle
            if self.lost_ball_counter > self.max_lost_frames:
                self.ball_history.clear()
                self.telemetry["status"] = "BALL LOST (RESET)"
            else:
                self.telemetry["status"] = f"BALL LOST ({self.lost_ball_counter}/{self.max_lost_frames})"
            return False
        
        # Top bulundu, sayacı sıfırla
        self.lost_ball_counter = 0

        # Cooldown Kontrolü
        if frame_idx - self.last_kick_frame < self.cooldown_frames:
            self.telemetry["status"] = "COOLDOWN"
            return False

        self.ball_history.append((ball_coord[0], ball_coord[1], frame_idx))
        if len(self.ball_history) < 3: 
            self.telemetry["status"] = "BUFFERING"
            return False

        # Fiziksel Vektörler
        p1, p2, p3 = list(self.ball_history)[-3:]
        v1 = np.array([p2[0]-p1[0], p2[1]-p1[1]])
        v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
        
        accel = np.linalg.norm(v2 - v1)
        v1_u = v1 / (np.linalg.norm(v1) + 1e-6)
        v2_u = v2 / (np.linalg.norm(v2) + 1e-6)
        dir_ch = np.dot(v1_u, v2_u)

        # Mesafe Kontrolü
        distances = [np.linalg.norm(ball_coord - p['coord']) for p in players]
        min_dist = min(distances) if distances else 99.0

        # Telemetriyi Güncelle
        self.telemetry.update({
            "accel": accel,
            "dir_change": dir_ch,
            "min_dist": min_dist,
            "status": "ARMED"
        })

        # Karar Mantığı ve Nedenleri
        is_physically_kick = (accel > self.accel_threshold or dir_ch < 0.3)
        is_near_enough = min_dist < self.dist_threshold

        if is_physically_kick and is_near_enough:
            self.last_kick_frame = frame_idx
            self.telemetry["status"] = "KICK!"
            return True
        else:
            # Neden tetiklenmediğini kaydet
            reasons = []
            if not is_physically_kick: reasons.append("LOW_ACCEL")
            if not is_near_enough: reasons.append("TOO_FAR")
            self.telemetry["reason"] = "|".join(reasons)
            
        return False