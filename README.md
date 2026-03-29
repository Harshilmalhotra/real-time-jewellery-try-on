# Real-Time AI Jewellery Try-On System

An AI-powered computer vision pipeline for high-precision virtual jewellery try-on, optimized for real-time mobile and web inference.

## 🚀 Project Overview
This system detects human ears and earlobes using a custom-trained **YOLOv8 Nano** model. It establishes a multi-stage perception flow:
1.  **Global Detection**: Locating the "wholeear" to define the spatial region of interest.
2.  **Point Precision**: Identifying the "earlobe" for accurate jewellery placement.
3.  **Optimization**: Leveraging **INT8 Quantization** and **ONNX Runtime** for sub-50ms latency on edge devices.

## 🏗️ System Architecture
-   **Backbone**: YOLOv8n (Nano)
-   **Resolution**: 512x512 pixels
-   **Accuracy (mAP50)**: ~96% (Trained on 5,500+ annotated images)
-   **Deployment**: Web, Android, and iOS via ONNX Runtime

## 🛠️ Engineering Solves
### 1. Dataset Refinement (Geometry-Based Restoration)
Initially, the dataset mapping was inconsistent. I implemented a **Geometry-Based Audit** script to re-classify detections programmatically:
-   Boxes with Area > 0.02 $\to$ `wholeear` (0)
-   Boxes with Area <= 0.02 $\to$ `earlobe` (1)
This restored 2,988 ear and 4,103 earlobe labels, ensuring the model could distinguish between context and landmarks.

### 2. High-Performance Inference
Designed for mobile-first deployment, the system uses dynamic quantization to reduce the model size from ~12MB to **~4MB**, significantly lowering memory overhead on mobile CPUs.

## 📦 Getting Started

### Prerequisites
```bash
pip install ultralytics opencv-python onnxruntime numpy
```

### Training
To reproduce the training with the refined dataset:
```bash
python3 train_model.py
```

### Live Webcam Inference
To run the live detection (optimized for mobile architecture):
```bash
python3 inference_webcam.py
```

## 📂 Project Structure
-   `train_model.py`: Training engine with custom augmentations (Mosaic, Mixup).
-   `inference_webcam.py`: Production-ready inference script using ONNX Runtime.
-   `restore_labels.py`: Dataset repair utility using spatial geometry.
-   `fix_labels.sh`: High-speed class refinement utility.
-   `runs/`: Directory containing training logs, plots, and ONNX weights.

## 📱 Future Deployment: React Native / Android
1.  Use the `best_quantized.onnx` weights.
2.  Integrate with `@onnxruntime/react-native`.
3.  Apply the internal **Letterbox Preprocessing** logic found in `inference_webcam.py` for aspect-ratio preservation.

---
*Created by Antigravity AI for Harshil Malhotra*
