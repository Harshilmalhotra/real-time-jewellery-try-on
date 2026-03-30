from ultralytics import YOLO
import os
import shutil

# Update this path to your best model
MODEL_PATH = "runs/detect/multi_feature_detection/v1_nano/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: {MODEL_PATH} not found.")
    exit()

print(f"🚀 Loading {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# Export to ONNX with High-Speed 320px (4x Faster)
print("📦 Exporting model to ONNX (320px, Opset 12)...")
# imgsz=320 is standard for real-time mobile/browser AI
path = model.export(format="onnx", imgsz=320, simplify=True, opset=12)

# Copy to root for easy web access
shutil.copy(path, "best.onnx")
print(f"✅ Export complete! 320px Model is ready at: {os.path.abspath('best.onnx')}")
