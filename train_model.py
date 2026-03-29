import os
import time
import torch
from ultralytics import YOLO
from onnxruntime.quantization import quantize_dynamic, QuantType

# =========================
# CONFIG
# =========================
DATA_PATH = "Earlobes.v11i.yolov8/data.yaml"
RUN_NAME = "ear_detector_refined"
EPOCHS = 150
IMG_SIZE = 512
BATCH_SIZE = 16   # for RTX 3050 (4GB) - adjusted for YOLOv8n


# =========================
# SYSTEM INFO
# =========================
def print_system_info():
    print("\n========== SYSTEM INFO ==========")
    cuda = torch.cuda.is_available()
    print(f"CUDA Available: {cuda}")

    if cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        print("⚠️ Using CPU")

    print("================================\n")


# =========================
# DATASET INFO
# =========================
def print_dataset_info():
    base = "Earlobes.v11i.yolov8"

    def count(path):
        return len(os.listdir(path)) if os.path.exists(path) else 0

    print("\n========== DATASET INFO ==========")
    print(f"Train Images: {count(f'{base}/train/images')}")
    print(f"Valid Images: {count(f'{base}/valid/images')}")
    print(f"Test  Images: {count(f'{base}/test/images')}")
    print("=================================\n")


# =========================
# TRAIN MODEL
# =========================
def train_model():
    print_system_info()
    print_dataset_info()

    device = 0 if torch.cuda.is_available() else "cpu"

    print("\n========== TRAIN CONFIG ==========")
    print(f"Epochs: {EPOCHS}")
    print(f"Image Size: {IMG_SIZE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Device: {'GPU' if device == 0 else 'CPU'}")
    print("Classes: wholeear + earlobe")
    print("=================================\n")

    model = YOLO("yolov8n.pt")

    start_time = time.time()

    model.train(
        data=DATA_PATH,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=device,
        name=RUN_NAME,

        single_cls=False,

        # augmentation
        augment=True,
        mosaic=1.0,
        mixup=0.2,
        degrees=15,
        scale=0.5,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        patience=50,  # early stopping
        workers=4,
        cache=True,
        verbose=True
    )

    end_time = time.time()

    print("\n========== TRAINING COMPLETE ==========")
    print(f"Total Time: {(end_time - start_time)/60:.2f} minutes")
    print("Best model saved automatically as best.pt")
    print("=======================================\n")


# =========================
# EXPORT + QUANTIZE
# =========================
def export_model():
    print("\n========== EXPORT ==========")

    import glob
    search_path = f"runs/detect/{RUN_NAME}*/weights/best.pt"
    matches = glob.glob(search_path)

    if not matches:
        print("❌ best.pt not found. Training failed.")
        return
        
    best_path = max(matches, key=os.path.getmtime)

    print(f"Loading model: {best_path}")

    model = YOLO(best_path)

    print("Exporting ONNX...")
    model.export(format="onnx", imgsz=IMG_SIZE, opset=12)

    onnx_path = best_path.replace(".pt", ".onnx")
    quant_path = best_path.replace(".pt", "_quantized.onnx")

    print("Quantizing model...")
    quantize_dynamic(
        onnx_path,
        quant_path,
        weight_type=QuantType.QInt8
    )

    print("\n========== EXPORT COMPLETE ==========")
    print(f"ONNX model: {onnx_path}")
    print(f"Quantized model: {quant_path}")
    print("=====================================\n")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    train_model()
    export_model()