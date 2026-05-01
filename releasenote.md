# v2.0.0 - Glimmer Studio: The Web Revolution ✨
**@Harshilmalhotra Harshilmalhotra released this now**

## v2.0.0 - Web-Native AR, Multimodal Fusion, and Necklace Support
This major release marks the transition from a Python-centric prototype to a full-scale **Web-Native AR Studio**. Introducing "Glimmer Studio", v2.0 brings hardware-accelerated browser inference, multimodal AI fusion, and expanded jewellery categories.

### 🌟 Major Highlights in v2.0
*   **Glimmer Studio (Web App):** A professional, browser-based AR interface that runs entirely on the client side. No server-side processing required.
*   **Multimodal AI Fusion:** A new hybrid detection engine that combines **YOLOv8** (for precision earlobe/ear detection) with **MediaPipe Pose** (for neck and shoulder tracking).
*   **Necklace Try-On Engine:** Leveraging the new fusion engine, v2.0 now supports real-time necklace placement with natural physics and body-anchor tracking.
*   **ONNX Runtime Web Integration:** High-performance hardware acceleration via WebGL/WebGPU, enabling 30+ FPS on modern browsers.
*   **Modular Architecture:** A completely refactored JavaScript core with dedicated modules for:
    *   `Detectors`: YOLO and Pose handlers.
    *   `LandmarkFusion`: Logic to merge coordinates from multiple AI models.
    *   `Render`: Advanced Canvas2D/WebGL rendering layers for jewellery.
*   **Dynamic Asset Studio:** Interactive UI to switch between different earring and necklace designs in real-time.

### 🛠️ Technical Stack (v2.0)
| Layer | Technology |
| :--- | :--- |
| **Inference Engine** | ONNX Runtime Web + MediaPipe |
| **Detection Models** | YOLOv8 Nano (Exported to ONNX) |
| **Landmark Tracking** | BlazePose (via MediaPipe) |
| **Rendering** | High-performance Canvas2D Modular Engine |
| **UI/UX** | Vanilla JS / CSS3 Glimmer Design System |

### 📂 Project Structure Changes
*   `src/glimmer-studio/`: The new home for the web-based AR application.
*   `src/glimmer-studio/src/`: Modular source code for detectors, fusion, and rendering.
*   `src/export_for_web.py`: Utility script to convert PyTorch models to optimized ONNX formats for the web.

### 🚀 How to Run Glimmer Studio
1. Navigate to the `src/glimmer-studio` directory.
2. Serve the directory using any local web server (e.g., `python -m http.server 8000`).
3. Open `localhost:8000` in your browser.
4. Click **"START STUDIO"** to wake the neural cores.

**Optimized for high-performance real-time web AR. Developed by Harshil Malhotra.**

---

### ⚖️ Licensing Information
Please note that this project operates under a **Custom Non-Commercial & Attribution License** (see `LICENSE.md` in the root directory). 
*   **Non-Commercial Use:** Free to use for personal, educational, and research purposes, provided that proper attribution is given.
*   **Commercial Use:** Any business-related use, monetization, or integration into commercial products requires a paid commercial license and royalty agreement. Please contact the developer for commercial licensing details.
