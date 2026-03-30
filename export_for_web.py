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

# Copy to Production and Development folders
shutil.copy(path, "best.onnx") 
shutil.copy(path, "glimmer-studio/best.onnx")

print(f"✅ Export complete! 320px Model is live in: {os.path.abspath('glimmer-studio/best.onnx')}")
