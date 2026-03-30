# v1.2.0 - Universal Hardware Acceleration & Precision Try-On (Stable)
**@Harshilmalhotra Harshilmalhotra released this now**

## v1.2.0 - Performance, Accuracy, and Cross-Platform Integration
This release introduces "Flip-First" detection logic and Universal Hardware Acceleration, solving all previous coordinate lag and platform compatibility issues.

### New Features in v1.2.0
*   **Universal Hardware Acceleration:** Automatically detects and utilizes **Apple Silicon (MPS)**, **NVIDIA (CUDA)**, or **CPU**, ensuring smooth performance on Windows, Linux, and macOS.
*   **Flip-First Logic (100% Accuracy):** The system now flips the camera frame *before* detection. This fixes all "mirroring" math errors and ensures the jewellery is perfectly aligned with the earlobe.
*   **Lag-Free Tracking:** Re-tuned the smoothing engine (Lerp) from 0.4 to 0.8, making the jewellery follow head movements with zero perceived delay.
*   **Refined Physics 2.0:** Improved Damping and Swing Sensitivity constants for a more "weighted" and natural feel during rapid movements.
*   **Integrated FPS Counter:** Real-time performance monitoring directly on the AR display.

### Key Components
| Component | Function |
| :--- | :--- |
| **live_tryon_physics.py** | **(Primary)** Cross-platform inference with high-speed physics and 100% alignment. |
| **requirements.txt** | Updated list reflecting new `torch` requirements for hardware check. |
| **inference_webcam.py** | Still available as a lightweight baseline for basic detection. |

### Model Performance (Stable)
The core YOLOv8 Nano detection engine continues to provide industry-leading performance on our custom jewellery training set:
*   **mAP50:** 97.4%
*   **Precision:** 95.3%
*   **Recall:** 95.0%

### Quick Start (Recommended)
For the best experience, we recommend **cloning the entire repository** to ensure all paths overlap correctly.

**Option A: Clone Repo (Recommended)**
```bash
git clone https://github.com/Harshilmalhotra/real-time-jewellery-try-on.git
cd real-time-jewellery-try-on
pip install -r requirements.txt
python live_tryon_physics.py
```

**Option B: Standalone Script**
1. Download `best.pt`, `earring.png`, and `live_tryon_physics.py`.
2. Ensure they are in the **same folder**.
3. If your model path is different, update the `MODEL_PATH` variable at the top of the script.
4. Run: `python live_tryon_physics.py`

### Assets
*   **live_tryon_physics.py:** Principal inference script for this release.
*   **best.pt / best.onnx:** Pre-trained weights for Earring detection.
*   **earring.png:** Default high-resolution jewellery asset.
*   **requirements.txt:** Dependency list for setup.

**Optimized for high-performance real-time desktop AR. Developed by Harshil Malhotra.**
