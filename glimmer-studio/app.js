/**
 * ✨ Glimmer AI - Virtual Jewellery Try-On (v2.0.0 WEB)
 * Developed by Harshil Malhotra
 * 
 * This uses ONNX Runtime Web to run YOLOv8 directly in the browser.
 * Zero-Lag Mirror with Background AI Processing.
 */

const CONFIG = {
    modelPath: 'best.onnx',
    earringImgPath: 'earring.png',
    confThreshold: 0.45,
    classes: ['earlobe', 'eye', 'nose', 'wholeear'],
    physics: {
        gravity: 0.9,
        damping: 0.93,
        swing: 0.18,
        stiffness: 0.06
    }
};

class EarringTracker {
    constructor(x, y, size) {
        this.x = x; this.y = y; this.size = size;
        this.angle = 0; this.vel = 0;
        this.lastX = x;
        this.active = true;
        this.missedFrames = 0;
    }

    update(tx, ty, ts, detected = true) {
        if (detected) {
            this.missedFrames = 0;
            this.x = this.x * 0.15 + tx * 0.85; 
            this.y = this.y * 0.15 + ty * 0.85;
            this.size = this.size * 0.6 + ts * 0.4;
            
            let dx = this.x - this.lastX;
            this.vel += -dx * CONFIG.physics.swing;
            this.lastX = this.x;
        } else {
            this.missedFrames++;
            if (this.missedFrames > 15) this.active = false;
        }

        let torque = -CONFIG.physics.gravity * Math.sin(this.angle * Math.PI / 180);
        let restoring = -this.angle * CONFIG.physics.stiffness;
        this.vel = (this.vel + torque + restoring) * CONFIG.physics.damping;
        this.angle += this.vel;
    }
}

let session = null;
let trackers = [];
let earringImg = new Image();
let lastDetections = [];
let isProcessingAI = false;

const canvas = document.getElementById('output-canvas');
const ctx = canvas.getContext('2d');
const video = document.getElementById('webcam');
const statusVal = document.getElementById('status-val');
const fpsVal = document.getElementById('fps-val');
const loadingOverlay = document.getElementById('loading-overlay');

// Global processing canvas at 320px for High-Speed AI
const offscreen = new OffscreenCanvas(320, 320);
const tCtx = offscreen.getContext('2d');

async function init() {
    try {
        statusVal.innerText = 'CALIBRATING PRECISION AI...';
        
        // Use v1.14.0 - Perfectly compatible with standard local servers
        ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/';
        ort.env.wasm.numThreads = 1; 
        ort.env.wasm.proxy = false;

        session = await ort.InferenceSession.create(CONFIG.modelPath, {
            executionProviders: ['wasm'], // Use WASM (CPU)
            graphOptimizationLevel: 'all'
        });

        await new Promise((res, rej) => {
            earringImg.onload = res;
            earringImg.onerror = rej;
            earringImg.src = CONFIG.earringImgPath;
        });

        statusVal.innerText = 'WAKING ROYAL MIRROR...';
        video.srcObject = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: 'user' }
        });
        await video.play();
        
        canvas.width = 640;
        canvas.height = 480;
        loadingOverlay.style.opacity = '0';
        setTimeout(() => loadingOverlay.style.display = 'none', 500);
        
        statusVal.innerText = 'ROYAL CORE ONLINE';
        
        // START THE LOOPS
        drawMirror(); // Smooth Display
        aiLoop();     // Background AI
    } catch (err) {
        console.error(err);
        statusVal.innerText = 'SYSTEM ERROR: ' + err.message;
    }
}

// 1. MIRROR LOOP (Locked to Screen Refresh Rate)
function drawMirror() {
    if (video.readyState < 2) return requestAnimationFrame(drawMirror);

    // Draw Mirror Feed
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    ctx.restore();

    // Match Physics using LAST detections
    let used = new Set();
    trackers.forEach(t => {
        let bestIdx = -1;
        let bestDist = 80;
        lastDetections.forEach((det, i) => {
            if (CONFIG.classes[det.label] !== 'earlobe') return;
            let dist = Math.hypot(t.x - det.cx, t.y - det.cy);
            if (!used.has(i) && dist < bestDist) {
                bestIdx = i; bestDist = dist;
            }
        });

        if (bestIdx !== -1) {
            let det = lastDetections[bestIdx];
            t.update(det.cx, det.cy, det.w, true);
            used.add(bestIdx);
        } else {
            t.update(0,0,0, false);
        }
    });

    lastDetections.forEach((det, i) => {
        if (!used.has(i) && CONFIG.classes[det.label] === 'earlobe') {
            trackers.push(new EarringTracker(det.cx, det.cy, det.w));
        }
    });
    trackers = trackers.filter(t => t.active);

    // 3. Render Tracking (Jewellery Only)
    trackers.forEach(t => {
        if (t.missedFrames > 4) return;
        
        // --- RENDER JEWELLERY ---
        ctx.save();
        ctx.translate(t.x, t.y);
        ctx.rotate(t.angle * Math.PI / 180);
        let scale = (t.size * 2.8) / earringImg.width;
        let ew = earringImg.width * scale;
        let eh = earringImg.height * scale;
        ctx.drawImage(earringImg, -ew/2, 0, ew, eh);
        ctx.restore();
    });

    requestAnimationFrame(drawMirror);
}

// 2. ASYNC AI LOOP
async function aiLoop() {
    if (isProcessingAI || video.readyState < 2) {
        return setTimeout(aiLoop, 10);
    }
    
    isProcessingAI = true;
    try {
        const start = performance.now();
        lastDetections = await runInference();
        fpsVal.innerText = Math.round(1000 / (performance.now() - start));
    } catch (err) {
        console.warn('AI skipping frame', err);
    }
    isProcessingAI = false;
    setTimeout(aiLoop, 5);
}

async function runInference() {
    // 1. Efficient Preprocessing at 320x320 (High-Speed)
    ctx.drawImage(video, 0, 0, 640, 480); // Ensure mirror is updated
    
    tCtx.save();
    tCtx.scale(-0.5, 0.5); // Resize 640 down to 320
    tCtx.drawImage(video, -640, 0, 640, 480);
    tCtx.restore();
    
    // We only need a 320x320 chunk for the High-Speed AI
    const imgData = tCtx.getImageData(0, 0, 320, 320).data;
    const input = new Float32Array(3 * 320 * 320);
    const area = 320 * 320;
    
    for (let i = 0; i < area; i++) {
        const i4 = i * 4;
        input[i] = imgData[i4] / 255.0;
        input[i + area] = imgData[i4 + 1] / 255.0;
        input[i + area * 2] = imgData[i4 + 2] / 255.0;
    }

    const tensor = new ort.Tensor('float32', input, [1, 3, 320, 320]);
    const results = await session.run({ images: tensor });
    const output = results[Object.keys(results)[0]].data;
    
    // 2. High-Speed Detection Filter
    const detections = [];
    const boxes = 2100; // YOLOv8n has fewer boxes (2100) at 320x320
    
    for (let i = 0; i < boxes; i++) {
        let maxConf = 0, clsIdx = -1;
        for (let j = 0; j < 4; j++) {
            let conf = output[(4 + j) * boxes + i];
            if (conf > maxConf) { maxConf = conf; clsIdx = j; }
        }

        if (maxConf > CONFIG.confThreshold) {
            // SCALE BACK TO 640x480 (Multiply by 2 for the 320->640 stretch)
            detections.push({
                cx: output[i] * 2, 
                cy: output[boxes + i] * 2,
                w: output[2 * boxes + i] * 2, 
                label: clsIdx
            });
        }
    }
    return detections;
}

document.getElementById('start-btn').onclick = init;
