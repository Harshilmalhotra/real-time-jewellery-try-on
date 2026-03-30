# ✨ Optimized Real-Time Jewellery Try-On with Physics (v1.2.0)
# Fixes lag, accuracy, and improves physics simulation

import cv2
import numpy as np
import time
import os
import torch # Added for cross-platform hardware detection
from ultralytics import YOLO


# =========================
# CONFIG
# =========================
MODEL_PATH = "runs/detect/multi_feature_detection/v1_nano/weights/best.pt"
CONF_THRESHOLD = 0.30 
SHOW_DEBUG = False # Toggle for clean look
EARRING_IMG_PATH = "earring.png"
CLASSES = ['earlobe', 'eye', 'nose', 'wholeear']

# Physics constants (Tuned for better feel)
GRAVITY = 0.85
DAMPING = 0.94
SWING_SENSITIVITY = 0.15 # Reduced to prevent "nervous" shaking
REST_STIFFNESS = 0.05
MAX_ANGLE = 40

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
            # Higher "Follow" speed (0.8 instead of 0.4) reduces perceived lag
            self.x = self.x * 0.2 + tx * 0.8
            self.y = self.y * 0.2 + ty * 0.8
            self.size = self.size * 0.7 + ts * 0.3
            
            # Use movement to trigger swing
            dx = float(self.x - self.last_x)
            self.vel += -dx * SWING_SENSITIVITY
            self.last_x, self.last_y = self.x, self.y
        else:
            self.missed_frames += 1
            if self.missed_frames > 20: 
                self.active = False
        
        # Pendulum Physics
        torque = -GRAVITY * np.sin(np.radians(self.angle))
        restoring = -self.angle * REST_STIFFNESS
        self.vel = (self.vel + torque + restoring) * DAMPING
        self.angle += self.vel
        
        if abs(self.angle) > MAX_ANGLE:
            self.angle = np.sign(self.angle) * MAX_ANGLE
            self.vel *= -0.4 # Bounce back slightly

def overlay_roi(background, overlay, M):
    """Optimized overlay with boundary checking"""
    h_bg, w_bg = background.shape[:2]
    h_ov, w_ov = overlay.shape[:2]
    
    # Calculate bounding box of the transformed overlay
    pts = np.array([[0,0], [w_ov,0], [w_ov,h_ov], [0,h_ov]], dtype='float32')
    t_pts = cv2.transform(np.array([pts]), M)[0]
    
    x1, y1 = int(max(0, np.min(t_pts[:, 0]))), int(max(0, np.min(t_pts[:, 1])))
    x2, y2 = int(min(w_bg, np.max(t_pts[:, 0]))), int(min(h_bg, np.max(t_pts[:, 1])))
    
    if x2 <= x1 or y2 <= y1: return

    # Correct Matrix for the ROI crop
    M_roi = M.copy()
    M_roi[0, 2] -= x1
    M_roi[1, 2] -= y1
    
    # Warp directly into small crop for speed
    warped = cv2.warpAffine(overlay, M_roi, (x2-x1, y2-y1), 
                            flags=cv2.INTER_LINEAR, 
                            borderMode=cv2.BORDER_CONSTANT, 
                            borderValue=(0,0,0,0))
    
    alpha = warped[:, :, 3:4] / 255.0
    fg_rgb = warped[:, :, :3]
    bg_roi = background[y1:y2, x1:x2]
    
    # Perform blending
    background[y1:y2, x1:x2] = (bg_roi * (1 - alpha) + fg_rgb * alpha).astype(np.uint8)

def main():
    print("🚀 Initializing Cross-Platform Physics Try-On...")
    
    # Universal Hardware Detection
    if torch.cuda.is_available():
        device = 'cuda' # NVIDIA GPUs (Windows/Linux)
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = 'mps'  # Apple Silicon (Mac)
    else:
        device = 'cpu'  # Fallback for everyone else
        
    print(f"Using compute device: {device}")
    
    model = YOLO(MODEL_PATH).to(device)
    
    earring_img = cv2.imread(EARRING_IMG_PATH, cv2.IMREAD_UNCHANGED)
    if earring_img is None: 
        return print(f"❌ Error: {EARRING_IMG_PATH} not found")
    
    # Earring pivot at top-center of image
    eh, ew = earring_img.shape[:2]
    pivot_img = (ew // 2, 0)
    
    cap = cv2.VideoCapture(0)
    trackers = []
    prev_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        # 1. FLIP FIRST - EVERYTHING DETECTED HERE WILL MATCH DISPLAY 100%
        frame = cv2.flip(frame, 1)
        
        # 2. RUN INFERENCE on Flipped Frame
        # Resizing internal YOLO buffer to 640 for speed while keeping raw frame resolution
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False, imgsz=640)[0]
        
        current_dets = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            if CLASSES[cls_id] == 'earlobe':
                x1, y1, x2, y2 = map(float, box.xyxy[0].tolist())
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                size = x2 - x1
                current_dets.append((cx, cy, size))

        # 3. TRACKING & PHYSICS
        used = set()
        new_trackers = []
        for t in trackers:
            best_idx = -1
            best_dist = 60 # Detection radius
            for i, det in enumerate(current_dets):
                dist = np.hypot(t.x-det[0], t.y-det[1])
                if i not in used and dist < best_dist:
                    best_idx = i; best_dist = dist
            
            if best_idx != -1:
                t.update(current_dets[best_idx][0], current_dets[best_idx][1], current_dets[best_idx][2], True)
                used.add(best_idx)
                new_trackers.append(t)
            else:
                t.update(0,0,0, False)
                if t.active: new_trackers.append(t)
        
        # Add new detection objects
        for i, det in enumerate(current_dets):
            if i not in used: 
                new_trackers.append(EarringPhysics(det[0], det[1], det[2]))
        trackers = new_trackers

        # 4. RENDER
        for t in trackers:
            if t.missed_frames > 3: continue
            
            # Earring Scale Factor (Tweak 2.5 to adjust earring size)
            scale = (t.size * 2.8) / ew
            
            # Calculate rotation matrix around the pivot point
            M = cv2.getRotationMatrix2D(pivot_img, t.angle, scale)
            
            # Translate to the detected earlobe center
            # Correcting for the rotation's internal translation
            c_px = M[0, 0] * pivot_img[0] + M[0, 1] * pivot_img[1] + M[0, 2]
            c_py = M[1, 0] * pivot_img[0] + M[1, 1] * pivot_img[1] + M[1, 2]
            M[0, 2] += (t.x - c_px)
            M[1, 2] += (t.y - c_py)
            
            overlay_roi(frame, earring_img, M)

        # 5. UI & FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {int(fps)} | Tracks: {len(trackers)}", (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Optimized Physics Try-On", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
