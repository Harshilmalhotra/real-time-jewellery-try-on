import cv2
import numpy as np
import onnxruntime as ort
import time

# =========================
# CONFIG
# =========================
MODEL_PATH = "runs/detect/ear_detector_refined/weights/best_quantized.onnx"
CONF_THRESHOLD = 0.45
IOU_THRESHOLD = 0.45
IMG_SIZE = 512
CLASSES = ['wholeear', 'earlobe']
COLORS = [(0, 255, 0), (0, 0, 255)]  # Green for Ear, Red for Earlobe

# =========================
# INFERENCE CLASS
# =========================
class EarDetector:
    def __init__(self, model_path):
        self.session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        self.input_name = self.session.get_inputs()[0].name
        print(f"Loaded model from {model_path}")

    def preprocess(self, frame):
        # Letterbox/Resize to 512x512
        h, w = frame.shape[:2]
        scale = min(IMG_SIZE / h, IMG_SIZE / w)
        new_h, new_w = int(h * scale), int(w * scale)
        
        resized = cv2.resize(frame, (new_w, new_h))
        
        # Create canvas and center image
        canvas = np.full((IMG_SIZE, IMG_SIZE, 3), 114, dtype=np.uint8)
        canvas[(IMG_SIZE - new_h)//2 : (IMG_SIZE - new_h)//2 + new_h, 
               (IMG_SIZE - new_w)//2 : (IMG_SIZE - new_w)//2 + new_w, :] = resized
        
        # BGR to RGB, Normalize, HWC to CHM
        input_data = canvas.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
        
        return input_data, scale, (IMG_SIZE - new_w)//2, (IMG_SIZE - new_h)//2

    def postprocess(self, outputs, scale, pad_w, pad_h, orig_shape):
        # Output shape: (1, 6, 5376) -> (6, 5376)
        predictions = np.squeeze(outputs).T
        
        boxes = []
        scores = []
        class_ids = []

        for pred in predictions:
            # pred: [cx, cy, w, h, cls0, cls1]
            box = pred[:4]
            cls_scores = pred[4:]
            class_id = np.argmax(cls_scores)
            score = cls_scores[class_id]
            
            if score > CONF_THRESHOLD:
                cx, cy, w, h = box
                
                # Convert from center to xyxy
                x1 = (cx - w/2 - pad_w) / scale
                y1 = (cy - h/2 - pad_h) / scale
                x2 = (cx + w/2 - pad_w) / scale
                y2 = (cy + h/2 - pad_h) / scale
                
                boxes.append([int(x1), int(y1), int(x2-x1), int(y2-y1)])
                scores.append(float(score))
                class_ids.append(class_id)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, IOU_THRESHOLD)
        
        results = []
        if len(indices) > 0:
            for i in indices.flatten():
                results.append({
                    "box": boxes[i],
                    "score": scores[i],
                    "class_id": class_ids[i]
                })
        return results

# =========================
# MAIN LOOP
# =========================
def main():
    detector = EarDetector(MODEL_PATH)
    cap = cv2.VideoCapture(0)  # Use 0 for primary webcam

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Starting Live Detection... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()

        # Inference
        input_data, scale, pad_w, pad_h = detector.preprocess(frame)
        outputs = detector.session.run(None, {detector.input_name: input_data})[0]
        results = detector.postprocess(outputs, scale, pad_w, pad_h, frame.shape)

        # Draw results
        for res in results:
            x, y, w, h = res["box"]
            cls_id = res["class_id"]
            score = res["score"]
            
            color = COLORS[cls_id]
            label = f"{CLASSES[cls_id]} {score:.2f}"
            
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # FPS calculation
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Jewellery Try-On: Ear Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
