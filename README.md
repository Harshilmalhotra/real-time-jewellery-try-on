# ✨ Real-Time Jewellery Try-On (Ear & Face Detection)

A high-performance, lightweight computer vision pipeline designed for real-time jewellery AR applications. This project provides a robust YOLOv8-based model capable of detecting critical facial and ear features with **>97% accuracy (mAP50)**.

---

## 🚀 Overview
This repository contains a complete pipeline for training and deploying feature detection models specifically optimized for **earlobes, eyes, noses, and whole ears**. These benchmarks are essential for placing digital earrings, necklaces, or facial accessories in real-time.

### 🎯 Supported Classes
- `earlobe`: High-precision detection for earring placement.
- `wholeear`: General ear bounding for spatial awareness.
- `eye`: Reference points for facial alignment.
- `nose`: Center point for symmetrical accessory orientation.

---

## 📦 Model Ecosystem
We provide several model formats to ensure seamless integration across different platforms:

| Format | Extension | Target Platform | Performance Focus |
| :--- | :--- | :--- | :--- |
| **PyTorch** | `.pt` | Research / Desktop | Native accuracy and training |
| **ONNX** | `.onnx` | PC / Web / Server | Cross-platform CPU/GPU speed |
| **TFLite** | `.tflite` | Android / IoT | Quantized for mobile edge devices |
| **CoreML** | `.mlpackage` | iOS / macOS | Native Apple Neural Engine (ANE) |

---

## ⚡ Quick Test (Standalone)
Don't want to clone the whole repo? You can test the model with just two files:

1. **Download the weights**: Grab `best.pt` from the `runs/detect/.../weights` folder.
2. **Download the script**: Grab `model_standalone_run.py`.
3. **Place both in the SAME folder** and run:
   ```bash
   pip install ultralytics opencv-python
   python model_standalone_run.py
   ```

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd real-time-jewellery-try-on
   ```

2. **Set up Environment**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install ultralytics opencv-python onnxruntime
   ```

---

## 🏃 Usage

### 1. Training from Scratch
Run the optimized training script. It handles data verification, training (100 epochs), and automatic multi-platform export.
```bash
python train_model.py
```

### 2. Real-Time Webcam Inference
Test the trained model immediately using your PC webcam.
```bash
python inference_webcam.py
```

---

## 📈 Model Performance
Based on the latest training run of 100 epochs on a dataset of **11,000+ images**:

- **mAP50 (Mean Average Precision):** 97.4%
- **mAP50-95:** 63.2%
- **Precision:** 95.3%
- **Recall:** 95.0%

> [!NOTE]
> Detailed training charts (Loss, Precision, R-curves) are available in the `runs/detect/multi_feature_detection/v1_nano/` directory.

---

## 🏗️ Project Structure
```text
.
├── train_model.py         # Automated training & export pipeline
├── inference_webcam.py    # Real-time CV2 detector (PC)
├── Earlobes.v11i.yolov8/  # Dataset and annotations
├── runs/detect/           # Training logs and multi-format weights
│   └── .../weights/       # best.pt, best.onnx, best.tflite, best.mlpackage
└── yolov8n.pt             # Base pre-trained model
```

## 🤝 Contributing
Feel free to fork this project and submit pull requests for hardware-specific optimizations or new feature detections.

---
*Developed by Harshil Malhotra*
