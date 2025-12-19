import cv2
import numpy as np
import os
from inference import InferenceEngine
from geometry import GeometryEngine
from kick_detector import KickDetector
from logic import (
    ProfessionalTeamClassifier, OffsideLogic, 
    FinalDecisionLogic, TemporalTeamVoter
)

class AutonomousVAR:
    def __init__(self, video_path, output_path="autonomous_var_output_13.mp4"):
        self.cap = cv2.VideoCapture(video_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        # Motorlar
        self.inference = InferenceEngine({
            'field': 'football-field-detection.pt',
            'player': 'football-players-detection.pt',
            'ball': 'football-ball-detection.pt'
        })
        self.geometry = GeometryEngine()
        self.team_classifier = ProfessionalTeamClassifier()
        self.team_voter = TemporalTeamVoter()
        self.kick_detector = KickDetector(fps=self.fps)
        self.offside_logic = OffsideLogic(fps=self.fps)
        self.final_logic = FinalDecisionLogic()

        self.frame_idx = 0
        self.freeze_frames_left = 0
        self.freeze_frame = None
        self.TEAM_COLORS = {0: (255, 50, 50), 1: (50, 50, 255), -1: (200, 200, 200)}

    def detect_collisions(self, tids, boxes, threshold=0.20):
        """IoU tabanlı kilit açma mekanizması."""
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                b1, b2 = boxes[i], boxes[j]
                x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
                x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
                if x2 < x1 or y2 < y1: continue
                inter = (x2 - x1) * (y2 - y1)
                area1 = (b1[2]-b1[0])*(b1[3]-b1[1])
                area2 = (b2[2]-b2[0])*(b2[3]-b2[1])
                # IoU: $IoU = \frac{Area(A \cap B)}{Area(A \cup B)}$
                if inter / min(area1, area2) > threshold:
                    self.team_voter.unlock_id(tids[i])
                    self.team_voter.unlock_id(tids[j])

    def run(self):
        print(f"[INFO] İşlem başlatıldı. Telemetri Aktif.")
        while self.cap.isOpened():
            if self.freeze_frames_left > 0:
                self.freeze_frames_left -= 1
                self.writer.write(self.freeze_frame)
                cv2.imshow('PRO-OFFSIDE VAR', self.freeze_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break
                continue

            ret, frame = self.cap.read()
            if not ret: break
            self.frame_idx += 1

            # 1. Inference & Geometry
            f_res = self.inference.predict_field(frame)
            if f_res.keypoints is not None:
                self.geometry.solve_from_model(f_res.keypoints.xy[0].cpu().numpy(), 
                                               f_res.keypoints.conf[0].cpu().numpy())

            p_res = self.inference.predict_players(frame)
            b_res = self.inference.predict_ball(frame)

            # 2. Entity Handling
            players, ball_m = self.process_entities(frame, p_res, b_res)

            # 3. Gatekeeper & Logic
            if self.team_classifier.is_trained:
                self.offside_logic.update_attacking_team(players, ball_m)
                smooth_line = self.offside_logic.calculate_offside_line(players)
                is_kick = self.kick_detector.analyze(self.frame_idx, ball_m, players)

                # Lag Compensation & VAR Decision
                if is_kick and smooth_line:
                    decision_line = self.offside_logic.raw_line_x
                    attackers = [p for p in players if p['team'] == self.offside_logic.attacking_team_id]
                    res = self.final_logic.process_decision(attackers, decision_line, self.offside_logic.attack_direction)
                    
                    self.freeze_frame = frame.copy()
                    self.render_var_decision(self.freeze_frame, res, decision_line)
                    self.freeze_frames_left = int(self.fps * 3)

                # UI Çizimleri (Dashboard buraya eklendi)
                self.render_telemetry_dashboard(frame, smooth_line)
                self.draw_standard_overlay(frame, players, smooth_line)
            else:
                # Kalibrasyon devam ediyorsa sadece ilerlemeyi göster
                cv2.putText(frame, f"CALIBRATING TEAMS: {len(self.team_classifier.calibration_buffer)}/100", 
                            (50, 50), 0, 0.8, (0, 255, 255), 2)

            self.writer.write(frame)
            cv2.imshow('PRO-OFFSIDE VAR', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        
        self.cap.release(); self.writer.release(); cv2.destroyAllWindows()

    def process_entities(self, frame, p_res, b_res):
        players = []
        if p_res.boxes.id is not None:
            tids, cids, boxes = p_res.boxes.id.int().cpu().tolist(), p_res.boxes.cls.int().cpu().tolist(), p_res.boxes.xyxy.int().cpu().tolist()
            self.detect_collisions(tids, boxes)
            for tid, cid, box in zip(tids, cids, boxes):
                if cid == 3: continue
                p_m = self.geometry.pixel_to_pitch(np.array([[(box[0]+box[2])/2, box[3]]]))
                if p_m.size == 0: continue
                if tid in self.team_voter.stable_teams:
                    t_id = self.team_voter.stable_teams[tid]
                else:
                    crop = frame[box[1]:box[3], box[0]:box[2]]
                    if self.team_classifier.is_trained:
                        t_id, _ = self.team_voter.vote(tid, self.team_classifier.predict(crop))
                    else:
                        self.team_classifier.calibrate_step(crop)
                        t_id = -1
                players.append({'team': t_id, 'coord': p_m[0], 'bbox': box, 'id': tid})
        
        ball_m = None
        if len(b_res.boxes) > 0:
            bx = b_res.boxes[0].xyxy[0].cpu().numpy()
            res = self.geometry.pixel_to_pitch(np.array([[(bx[0]+bx[2])/2, bx[3]]]))
            if res.size > 0: ball_m = res[0]
        return players, ball_m

    def render_telemetry_dashboard(self, frame, line_x):
        """Dashboard'un geri geldiği kısım."""
        tel = self.kick_detector.telemetry
        logic = self.offside_logic
        overlay = frame.copy()
        cv2.rectangle(overlay, (self.width-350, 10), (self.width-10, 300), (0,0,0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        px = self.width - 330
        info = [
            (f"FRAME: {self.frame_idx}", (255,255,255)),
            (f"ATTACK TEAM: T{logic.attacking_team_id}", (0,255,255)),
            (f"DIRECTION: {logic.attack_direction}", (0,255,255)),
            (f"OFFSIDE LINE: {line_x:.2f}m" if line_x else "N/A", (0,255,0)),
            ("-" * 20, (100,100,100)),
            (f"KICK STATUS: {tel['status']}", (255,0,255)),
            (f"ACCEL: {tel['accel']:.2f}", (255,255,255)),
            (f"MIN_DIST: {tel['min_dist']:.2f}m", (255,255,255))
        ]
        for i, (txt, clr) in enumerate(info):
            cv2.putText(frame, txt, (px, 40 + i*30), 0, 0.6, clr, 2)

    def draw_standard_overlay(self, frame, players, line_x):
        if line_x:
            pts = self.geometry.get_offside_line(line_x)
            if len(pts) == 2: cv2.line(frame, pts[0], pts[1], (0, 255, 255), 2)
        for p in players:
            x1, y1, x2, y2 = p['bbox']
            is_off = (self.offside_logic.attack_direction == "L2R" and p['coord'][0] > line_x) or \
                     (self.offside_logic.attack_direction == "R2L" and p['coord'][0] < line_x) if line_x and p['team'] == self.offside_logic.attacking_team_id else False
            color = (0,0,0) if is_off else self.TEAM_COLORS.get(p['team'], (200,200,200))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            if is_off:
                cv2.rectangle(frame, (x1, y1-20), (x1+65, y1), (0,0,0), -1)
                cv2.putText(frame, "OFFSIDE", (x1+2, y1-5), 0, 0.45, (255,255,255), 1)

    def render_var_decision(self, frame, res, line_x):
        color = (0,0,255) if res['is_offside'] else (0,255,0)
        cv2.rectangle(frame, (0,0), (self.width, 150), (0,0,0), -1)
        cv2.putText(frame, f"VAR DECISION: {'OFFSIDE' if res['is_offside'] else 'CLEAN'}", (50, 70), 0, 1.5, color, 4)
        cv2.putText(frame, f"Margin: {res['margin']:.2f}m", (50, 120), 0, 1.0, (255,255,255), 2)
        pts = self.geometry.get_offside_line(line_x)
        if len(pts) == 2: cv2.line(frame, pts[0], pts[1], color, 6)

if __name__ == "__main__":
    system = AutonomousVAR("video.mp4")
    system.run()