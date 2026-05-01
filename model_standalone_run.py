# Copyright (c) 2026 Harshil Malhotra. All rights reserved.
# This code is subject to the terms of the Custom Non-Commercial & Attribution License 
# found in the LICENSE.md file in the root directory of this source tree.
# Commercial use requires a paid license.

import cv2
from ultralytics import YOLO
import time
import os

# =========================
# STANDALONE CONFIG
# =========================
# This script expects 'best.pt' to be in the same folder.
# You can also change this to 'best.onnx' if you prefer.
MODEL_FILENAME = "best.pt" 

# Detection settings
CONF_THRESHOLD = 0.30
CLASSES = ['earlobe', 'eye', 'nose', 'wholeear']
COLORS = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (0, 0, 255)] # Green, Blue, Yellow, Red

def start_standalone_detection():
    # 1. Check for the model file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, MODEL_FILENAME)
    
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: '{MODEL_FILENAME}' not found in the current folder!")
        print(f"👉 Please place your trained model file in: {current_dir}\n")
        return

    # 2. Load the model
    print(f"🔄 Loading {MODEL_FILENAME}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ FAILED to load model: {e}")
        return

    # 3. Initialize Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ ERROR: Could not access the webcam.")
        return

    print("\n" + "="*40)
    print("✨ STANDALONE DETECTION ACTIVE")
    print("   Press 'q' to quit.")
    print("="*40 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret: break

        start_time = time.time()

        # 4. Predict
        # We use stream=True for better memory efficiency in some environments
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        # 5. Render detections
        for box in results.boxes:
            # Box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Metadata
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            # Visuals
            color = COLORS[cls_id] if cls_id < len(COLORS) else (255, 255, 255)
            label = f"{CLASSES[cls_id]} {conf:.2f}"

            # Drawing
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Performance overlay
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        # Display
        cv2.imshow("Standalone Jewellery Detection Overlay", frame)

        # Exit handler
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("👋 Detection stopped.")

if __name__ == "__main__":
    start_standalone_detection()
