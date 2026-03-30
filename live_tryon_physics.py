# ✨ Real-Time Jewellery Try-On with Physics (v1.1.0)
# Developed by Harshil Malhotra

import cv2
import numpy as np
import time
import os
from ultralytics import YOLO

# =========================
# CONFIG
# =========================
MODEL_PATH = "runs/detect/multi_feature_detection/v1_nano/weights/best.pt"
CONF_THRESHOLD = 0.30 
SHOW_DEBUG_BOXES = False # Set to False for production/clean look
EARRING_IMG_PATH = "earring.png"
CLASSES = ['earlobe', 'eye', 'nose', 'wholeear']

# Physics constants
GRAVITY = 0.8
DAMPING = 0.92
SWING_SENSITIVITY = 0.25
REST_STIFFNESS = 0.08
MAX_ANGLE = 45

class EarringPhysics:
    def __init__(self, x, y, size):
        self.x, self.y = x, y
        self.size = size
        self.angle, self.vel = 0.0, 0.0
        self.last_x, self.last_y = x, y
        self.missed_frames = 0
        self.active = True

    def update(self, tx, ty, ts, detected=True):
        if detected:
            self.missed_frames = 0
            # Position smoothing (Lerp)
            self.x = self.x * 0.6 + tx * 0.4
            self.y = self.y * 0.6 + ty * 0.4
            self.size = self.size * 0.8 + ts * 0.2
            
            # Physics impulse
            dx = float(self.x - self.last_x)
            self.vel += -dx * SWING_SENSITIVITY
            self.last_x, self.last_y = self.x, self.y
        else:
            self.missed_frames += 1
            if self.missed_frames > 15: 
                self.active = False
        
        # Pendulum Physics
        torque = -GRAVITY * np.sin(np.radians(self.angle))
        restoring = -self.angle * REST_STIFFNESS
        self.vel = (self.vel + torque + restoring) * DAMPING
        self.angle += self.vel
        
        if abs(self.angle) > MAX_ANGLE:
            self.angle = np.sign(self.angle) * MAX_ANGLE
            self.vel *= -0.3

def overlay_roi_optimized(background, overlay, M):
    h_bg, w_bg = background.shape[:2]
    h_ov, w_ov = overlay.shape[:2]
    pts = np.array([[0,0], [w_ov,0], [w_ov,h_ov], [0,h_ov]], dtype='float32')
    t_pts = cv2.transform(np.array([pts]), M)[0]
    x1, y1 = int(max(0, np.min(t_pts[:, 0]))), int(max(0, np.min(t_pts[:, 1])))
    x2, y2 = int(min(w_bg, np.max(t_pts[:, 0]))), int(min(h_bg, np.max(t_pts[:, 1])))
    if x2 <= x1 or y2 <= y1: return
    M_roi = M.copy()
    M_roi[0, 2] -= x1; M_roi[1, 2] -= y1
    warped = cv2.warpAffine(overlay, M_roi, (x2-x1, y2-y1), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))
    alpha = warped[:, :, 3:4] / 255.0
    fg_rgb = warped[:, :, :3]
    bg_roi = background[y1:y2, x1:x2]
    background[y1:y2, x1:x2] = (bg_roi * (1 - alpha) + fg_rgb * alpha).astype(np.uint8)

def main():
    print("Loading model...")
    model = YOLO(MODEL_PATH)
    earring_img = cv2.imread(EARRING_IMG_PATH, cv2.IMREAD_UNCHANGED)
    if earring_img is None: return print("Error: earring.png not found")
    eh, ew = earring_img.shape[:2]
    pivot_img = (ew // 2, 0)
    
    cap = cv2.VideoCapture(0)
    trackers = []
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # Detection on ORIGINAL frame for best accuracy
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        
        # Flip frame for display
        frame = cv2.flip(frame, 1)
        w_f = frame.shape[1]
        
        current_dets = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if CLASSES[cls_id] == 'earlobe':
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                # Mirror coordinates for the flipped frame
                # x_mirror = width - x_orig
                cx_orig = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                cx = w_f - cx_orig
                size = x2 - x1
                current_dets.append((cx, cy, size, (w_f-x2, y1, w_f-x1, y2)))

        # Matching Logic
        used = set()
        new_trackers = []
        for t in trackers:
            best_idx = -1
            for i, det in enumerate(current_dets):
                if i not in used and np.hypot(t.x-det[0], t.y-det[1]) < 100:
                    best_idx = i; break
            if best_idx != -1:
                t.update(current_dets[best_idx][0], current_dets[best_idx][1], current_dets[best_idx][2], True)
                used.add(best_idx)
                new_trackers.append(t)
            else:
                t.update(0,0,0, False)
                if t.active: new_trackers.append(t)
        
        for i, det in enumerate(current_dets):
            if i not in used: new_trackers.append(EarringPhysics(det[0], det[1], det[2]))
        trackers = new_trackers

        # Render
        for t in trackers:
            if t.missed_frames > 2: continue
            
            # Temporary Debug Box
            if SHOW_DEBUG_BOXES:
                # Find matching detection box for visualization
                for det in current_dets:
                    if np.hypot(t.x-det[0], t.y-det[1]) < 10:
                        dx1, dy1, dx2, dy2 = map(int, det[3])
                        cv2.rectangle(frame, (dx1, dy1), (dx2, dy2), (255, 255, 0), 1)

            # Draw Earring
            scale = (t.size * 3.5) / ew
            M = cv2.getRotationMatrix2D(pivot_img, t.angle, scale)
            c_px = M[0, 0] * pivot_img[0] + M[0, 1] * pivot_img[1] + M[0, 2]
            c_py = M[1, 0] * pivot_img[0] + M[1, 1] * pivot_img[1] + M[1, 2]
            M[0, 2] += (int(t.x) - c_px)
            M[1, 2] += (int(t.y) - c_py)
            overlay_roi_optimized(frame, earring_img, M)

        cv2.imshow("Physics Try-On", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__": main()
