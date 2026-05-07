# 📱 React Native Android Porting Guide: Jewellery Try-On

This guide outlines the steps required to migrate the **Real-Time Jewellery Try-On** features from the Python/OpenCV desktop environment to a **React Native Android** application.

---

## 1. Model Export (TFLite)
Android applications run YOLO models most efficiently using the **TensorFlow Lite (.tflite)** format.

### Exporting the Model
Run the following script in your Python environment to convert your trained weights:

```python
from ultralytics import YOLO

# Load your best trained model
model = YOLO("runs/detect/multi_feature_detection/v1_nano/weights/best.pt")

# Export to TFLite with metadata for Android
# imgsz=320 is recommended for real-time mobile performance
model.export(format="tflite", imgsz=320, int8=True)
```

**Output:** You will get a `best_int8.tflite` file. Move this to your React Native project's `android/app/src/main/assets/` directory.

---

## 2. Dependencies
You will need the following libraries in your React Native project:

```bash
# Core Camera & Inference
npm install react-native-vision-camera
npm install vision-camera-plugin-yolov8 # Or a custom TFLite bridge

# High-Performance Graphics
npm install @shopify/react-native-skia

# Reanimated for smooth UI transitions
npm install react-native-reanimated
```

---

## 3. The Vision Camera Setup
Use `react-native-vision-camera` to handle the real-time frame buffer.

```javascript
import { Camera, useCameraDevices, useFrameProcessor } from 'react-native-vision-camera';
import { scanYoloV8 } from 'vision-camera-plugin-yolov8'; // Example plugin

function TryOnScreen() {
  const devices = useCameraDevices();
  const device = devices.front; // Use front camera for earrings

  const frameProcessor = useFrameProcessor((frame) => {
    'worklet';
    // Run detection on each frame
    const objects = scanYoloV8(frame, {
      modelPath: 'best_int8.tflite',
      confidenceThreshold: 0.3
    });
    
    // Pass 'objects' to your UI/Physics thread
  }, []);

  if (device == null) return <LoadingView />;
  return (
    <Camera
      style={StyleSheet.absoluteFill}
      device={device}
      isActive={true}
      frameProcessor={frameProcessor}
    />
  );
}
```

---

## 4. Physics Engine (Ported to TypeScript)
Convert the `EarringPhysics` class from Python to TypeScript to handle the "swing" effect based on head movement.

```typescript
export class EarringPhysics {
  x: number; y: number; size: number;
  angle = 0; vel = 0; lastX: number;
  
  constructor(x: number, y: number, size: number) {
    this.x = x; this.y = y; this.size = size;
    this.lastX = x;
  }

  update(tx: number, ty: number, ts: number, detected = true) {
    if (detected) {
      // Smoothing (Lerp)
      this.x = this.x * 0.2 + tx * 0.8;
      this.y = this.y * 0.2 + ty * 0.8;
      this.size = this.size * 0.7 + ts * 0.3;
      
      const dx = this.x - this.lastX;
      this.vel += -dx * 0.15; // Swing sensitivity
      this.lastX = this.x;
    }
    
    // Pendulum Physics Simulation
    const gravity = 0.85;
    const damping = 0.94;
    const torque = -gravity * Math.sin(this.angle * (Math.PI / 180));
    const restoring = -this.angle * 0.05;
    
    this.vel = (this.vel + torque + restoring) * damping;
    this.angle += this.vel;
  }
}
```

---

## 5. Rendering with Skia
Use **React Native Skia** to draw the earrings. This is much smoother than standard React Native views.

```javascript
import { Canvas, Image, useImage, Group } from "@shopify/react-native-skia";

const EarringOverlay = ({ physics }) => {
  const image = useImage(require("./assets/earring.png"));
  if (!image) return null;

  return (
    <Canvas style={StyleSheet.absoluteFill}>
      <Group
        origin={{ x: physics.x, y: physics.y }}
        transform={[
          { translateX: physics.x },
          { translateY: physics.y },
          { rotate: physics.angle * (Math.PI / 180) },
          { scale: physics.size / 100 } // Adjust scale factor
        ]}
      >
        <Image 
          image={image} 
          x={-image.width() / 2} 
          y={0} // Pivot at top-center
          width={image.width()} 
          height={image.height()} 
        />
      </Group>
    </Canvas>
  );
};
```

---

## 6. Android Configuration Checklist

1.  **Permissions**:
    Add to `android/app/src/main/AndroidManifest.xml`:
    ```xml
    <uses-permission android:name="android.permission.CAMERA" />
    ```

2.  **Asset Handling**:
    Ensure `.tflite` files are included in `bundleInFree: true` or `assets` folder in your `build.gradle`.

3.  **Performance**:
    - Enable **TurboModules** and **Fabric** if using React Native 0.70+.
    - Use the **GPU Delegate** in your TFLite configuration for 30+ FPS.

---

## 7. Critical Logic: Coordinate Mapping & Threading

To ensure the earrings actually sit on the ears (and don't jitter), you need to handle the conversion from the model's 320x320 space to the device's screen space.

### A. Coordinate Mapping Math
YOLO outputs coordinates in a normalized `[0, 1]` range or a fixed `[0, 320]` range. You must scale these to the `Canvas` dimensions while maintaining the aspect ratio.

```typescript
/**
 * Maps a coordinate from the model's output space to the screen space.
 */
function mapToScreen(
  value: number, 
  modelSize: number, 
  screenSize: number, 
  isMirrored: boolean
) {
  'worklet';
  // 1. Normalize (0 to 1)
  let normalized = value / modelSize;

  // 2. Mirror if using front camera
  if (isMirrored) {
    normalized = 1 - normalized;
  }

  // 3. Scale to screen
  return normalized * screenSize;
}
```

### B. Aspect Ratio Handling (Letterboxing)
If your camera stream is 4:3 but your screen is 19.5:9, you need to account for the "black bars" or "cropping" applied by the `Camera` view.

```typescript
const scaleX = screenWidth / frameWidth;
const scaleY = screenHeight / frameHeight;
const scale = Math.max(scaleX, scaleY);

const offsetX = (screenWidth - frameWidth * scale) / 2;
const offsetY = (screenHeight - frameHeight * scale) / 2;

// Final Mapping
const screenX = modelX * (frameWidth / 320) * scale + offsetX;
```

### C. Jitter Reduction (Lerp)
To prevent the jewellery from "shaking" due to slight detection variations, use **Linear Interpolation (Lerp)**.

```typescript
function lerp(start: number, end: number, t: number) {
  'worklet';
  return start * (1 - t) + end * t;
}

// Inside your physics/render loop
// t = 0.2 (low value = high smoothing, but more lag)
currentX.value = lerp(currentX.value, targetX, 0.2);
```

### D. Shared Values (Threading)
**NEVER** use `useState` for the detection coordinates. It will cause the UI to lag. 

1. Use `useSharedValue` from `react-native-reanimated`.
2. Update them directly inside the `useFrameProcessor`.
3. Use `useDerivedValue` or a Skia `useFrameCallback` to read them and draw.

---

*Generated for Glimmer Studio / Jewellery Try-On AR Project*
