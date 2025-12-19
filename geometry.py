import cv2
import numpy as np
from typing import List, Tuple, Optional

class GeometryEngine:
    """
    Sektör Standardı: Dinamik anahtar nokta eşleşmeli Homografi Motoru.
    32 noktalı SoccerNet/FIFA model çıktılarını metrik koordinatlara dönüştürür.
    """
    def __init__(self):
        # FIFA Standart Saha Ölçüleri (Metre)
        self.PITCH_WIDTH = 105.0
        self.PITCH_HEIGHT = 68.0
        self.homography_matrix: Optional[np.ndarray] = None
        self.h_inv: Optional[np.ndarray] = None
        
        # Kesinleştirilmiş Metrik Mapping (X, Y)
        self.PITCH_KEYPOINTS = {
            0: [0, 0],           1: [0, 13.85],    2: [0, 24.85],    3: [0, 43.15],
            4: [0, 54.15],       5: [0, 68],       6: [5.5, 24.85],  7: [5.5, 43.15],
            8: [11, 34],         9: [16.5, 13.85], 10: [16.5, 26.69], 11: [16.5, 41.31],
            12: [16.5, 54.15],   13: [52.5, 0],    14: [52.5, 24.85], 15: [52.5, 43.15],
            30: [43.35, 34],     31: [61.65, 34],  16: [52.5, 68],    24: [105, 0],
            17: [88.5, 13.85],   20: [88.5, 54.15], 25: [105, 13.85], 21: [94, 34],
            18: [88.5, 26.69],   19: [88.5, 41.31], 22: [99.5, 24.85], 23: [99.5, 43.15],
            26: [105, 24.85],    27: [105, 43.15], 28: [105, 54.15],  29: [105, 68]
        }

    def solve_from_model(self, kpts_from_model: np.ndarray, confidences: np.ndarray) -> bool:
        """
        Gelişmiş Çözücü: Confidence filtering ve Outlier reddetme içerir.
        """
        src_pts, dst_pts = [], []
        
        CONF_THRESHOLD = 0.6 

        for idx, (kp, conf) in enumerate(zip(kpts_from_model, confidences)):
            u, v = kp
            if conf >= CONF_THRESHOLD and u > 0 and v > 0 and idx in self.PITCH_KEYPOINTS:
                src_pts.append([u, v])
                dst_pts.append(self.PITCH_KEYPOINTS[idx])
        
        if len(src_pts) >= 6:
            new_h = self.calculate_robust_h(np.array(src_pts), np.array(dst_pts))
            if new_h is not None:
                self.homography_matrix = new_h
                self.h_inv = np.linalg.inv(new_h)
                return True
        return False

    def calculate_robust_h(self, src_pts, dst_pts):
        """RANSAC ile gürültüyü temizler ve matrisi doğrular."""
        H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 3.0)
        if H is None: return None

        # Geometrik Doğrulama: Matrisin determinantı çok küçükse 
        # veya aşırı bükülme varsa reddet.
        det = np.linalg.det(H[:2, :2])
        if abs(det) < 1e-6: return None # Singüler veya çok bozuk matris
        
        return H

    def update_homography_dynamic(self, src_pts: np.ndarray, dst_pts: np.ndarray) -> bool:
        """
        RANSAC algoritması ile gürültülü veriyi temizleyerek matrisi hesaplar.
        """
        try:
            # RANSAC, hatalı tespit edilen (outlier) keypoint'leri eler.
            H, status = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                self.homography_matrix = H
                self.h_inv = np.linalg.inv(H)
                return True
        except Exception as e:
            print(f"[CRITICAL] Homography calculation failed: {e}")
        return False

    def pixel_to_pitch(self, points: np.ndarray) -> np.ndarray:
        """Piksel -> Metre dönüşümü."""
        if self.homography_matrix is None: return np.array([])
        
        points_homo = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        transformed = (self.homography_matrix @ points_homo.T).T
        return transformed[:, :2] / transformed[:, 2:3]

    def pitch_to_pixel(self, points: np.ndarray) -> np.ndarray:
        """Metre -> Piksel dönüşümü (Ofsayt çizgisi çizimi için)."""
        if self.h_inv is None: return np.array([])
            
        points_homo = np.concatenate([points, np.ones((len(points), 1))], axis=1)
        transformed = (self.h_inv @ points_homo.T).T
        return transformed[:, :2] / transformed[:, 2:3]

    def get_offside_line(self, offside_x_metric: float) -> List[Tuple[int, int]]:
        """Ofsayt X koordinatını (metre) görüntüdeki perspektif çizgiye çevirir."""
        pitch_line = np.array([
            [offside_x_metric, 0],
            [offside_x_metric, self.PITCH_HEIGHT]
        ], dtype=np.float32)
        
        pixel_line = self.pitch_to_pixel(pitch_line)
        if len(pixel_line) < 2: return []
        return [(int(p[0]), int(p[1])) for p in pixel_line]