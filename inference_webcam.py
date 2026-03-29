import cv2
from ultralytics import YOLO
import time
import os

# =========================
# CONFIG
# =========================
# You can use 'best.pt' for highest accuracy or 'best.onnx' for faster CPU speed
# The path matches your latest training run
MODEL_PATH = "runs/detect/multi_feature_detection/v1_nano/weights/best.pt" 
CONF_THRESHOLD = 0.30
CLASSES = ['earlobe', 'eye', 'nose', 'wholeear']
# Green, Blue, Yellow, Red
COLORS = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (0, 0, 255)]

def main():
    # 1. Load Model
    model_path = MODEL_PATH
    print(f"Loading model from {model_path}...")
    
    if not os.path.exists(model_path):
        # Fallback to ONNX if .pt is missing or vice versa
        alt_path = model_path.replace(".pt", ".onnx")
        if os.path.exists(alt_path):
            model_path = alt_path
        else:
            print(f"\n❌ Error: Model NOT found at {model_path}")
            print("👉 Please run 'python train_model.py' first to train the model.\n")
            return

    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"❌ Error: Could not load model. {e}")
        return

    # 2. Open Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open webcam.")
        return

    print("🚀 Live Detection Started! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        start_time = time.time()

        # 3. Run Inference
        # verbose=False keeps the terminal clean
        results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)[0]

        # 4. Draw Results
        for box in results.boxes:
            # Get coordinates [x1, y1, x2, y2]
            coords = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, coords)
            
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            
            color = COLORS[cls_id] if cls_id < len(COLORS) else (255, 255, 255)
            label = f"{CLASSES[cls_id]} {conf:.2f}"

            # Draw rectangle and text
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # FPS calculation
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        # Show feed
        cv2.imshow("Jewellery Try-On: 4-Class Detection", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
