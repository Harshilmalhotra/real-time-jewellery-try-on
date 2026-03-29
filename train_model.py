import os
import torch
import yaml
from ultralytics import YOLO

# Configuration
DATASET_YAML = "/home/harshil-malhotra/Desktop/real-time-jewellery-try-on/Earlobes.v11i.yolov8/data.yaml"
MODEL_TYPE = "yolov8n.pt"  # Nano model: Perfect for mobile (Android/iOS)
PROJECT_NAME = "multi_feature_detection"
EXPERIMENT_NAME = "v1_nano"

def train():
    # Load dataset info for summary
    with open(DATASET_YAML, 'r') as f:
        data_info = yaml.safe_load(f)
    classes = data_info.get('names', [])

    # Detect device
    device = 0 if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name(0) if device != 'cpu' else 'CPU'

    print("\n" + "="*50)
    print("🚀 TRAINING CONFIGURATION SUMMARY")
    print("="*50)
    print(f"🔹 Model Type    : {MODEL_TYPE}")
    print(f"🔹 Dataset       : {DATASET_YAML}")
    print(f"🔹 Classes ({len(classes)}) : {', '.join(classes)}")
    print(f"🔹 Image Size    : 640")
    print(f"🔹 Epochs        : 100")
    print(f"🔹 Device        : {device_name}")
    print(f"🔹 Project       : {PROJECT_NAME}/{EXPERIMENT_NAME}")
    
    if device == 'cpu':
        print("\n⚠️  WARNING: CUDA not detected. Training will be VERY SLOW on CPU.")
    else:
        print("\n✅ GPU Accelerating enabled.")
    print("="*50 + "\n")

    # 1. Load the pre-trained Nano model
    print(f"Loading {MODEL_TYPE}...")
    model = YOLO(MODEL_TYPE)

    # 2. Train the model
    # Key parameters for high accuracy and mobile readiness:
    # imgsz=640: Standard for YOLOv8, good balance of speed/accuracy
    # epochs=100: Sufficient for 11k images to reach >90% mAP
    # batch=-1: Auto-detection of best batch size for your GPU memory
    print("Starting training...")
    results = model.train(
        data=DATASET_YAML,
        epochs=100,
        imgsz=640,
        batch=-1,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        device=device,
        cache=True,    # Cache images for faster training
        patience=20,   # Early stopping if no improvement for 20 epochs
        save=True,
        exist_ok=True
    )

    print("Training complete!")
    print(f"Results saved to: {results.save_dir}")

    # 3. Export for Mobile
    best_model_path = os.path.join(results.save_dir, 'weights', 'best.pt')
    if os.path.exists(best_model_path):
        trained_model = YOLO(best_model_path)
        
        # Export to ONNX (for PC Inference)
        print("Exporting best model to ONNX...")
        trained_model.export(format='onnx', opset=12) 
        
        # Export to TFLite (Android)
        print("Exporting best model to TFLite for Android...")
        trained_model.export(format='tflite', int8=True)  # int8 quantization for speed
        
        # Export to CoreML (iOS)
        print("Exporting best model to CoreML for iOS...")
        trained_model.export(format='coreml', nms=True)   # Include NMS for easier device integration
        
        print(f"Mobile models exported to: {os.path.dirname(best_model_path)}")
    else:
        print(f"Error: Could not find best model at {best_model_path}")

if __name__ == "__main__":
    train()
