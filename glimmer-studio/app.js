/**
 * ✨ Glimmer AR Engine v3.0 (MODULAR)
 * Integrated YOLOv8 + MediaPipe Pose Fusion
 */

import { YoloDetector } from './src/detectors/yoloDetector.js';
import { PoseDetector } from './src/detectors/poseDetector.js';
import { LandmarkFusion } from './src/fusion/landmarkFusion.js';
import { EarringEngine } from './src/jewellery/earringEngine.js';
import { NecklaceRenderer } from './src/render/necklaceRenderer.js';

const CONFIG = {
    yolo: {
        modelPath: 'best.onnx',
        confThreshold: 0.45
    },
    earringImg: 'earring.png',
    necklaceImg: 'necklace_1.png'
};

class GlimmerApp {
    constructor() {
        this.video = document.getElementById('webcam');
        this.canvas = document.getElementById('output-canvas');
        this.ctx = this.canvas.getContext('2d');
        
        this.yolo = new YoloDetector(CONFIG.yolo);
        this.pose = new PoseDetector();
        this.fusion = new LandmarkFusion();
        
        this.earrings = new EarringEngine(this.ctx);
        this.necklace = new NecklaceRenderer(this.ctx);
        
        this.state = {
            showEarrings: true,
            showNecklace: true,
            isAIProcessing: false,
            lastYoloDetections: [],
            lastPoseData: null,
            lastFusion: null
        };

        this.ui = {
            status: document.getElementById('status-val'),
            fps: document.getElementById('fps-val'),
            overlay: document.getElementById('loading-overlay'),
            startBtn: document.getElementById('start-btn'),
            toggleEarrings: document.getElementById('toggle-earrings'),
            toggleNecklace: document.getElementById('toggle-necklace')
        };
    }

    async init() {
        try {
            this.ui.status.innerText = 'INITIALIZING NEURAL CORES...';
            
            // 1. Initialize Detectors
            await Promise.all([
                this.yolo.init(),
                this.pose.init()
            ]);

            // 2. Load Assets
            await Promise.all([
                this.earrings.loadAsset(CONFIG.earringImg),
                this.necklace.loadAsset(CONFIG.necklaceImg)
            ]);

            // 3. Setup Webcam
            this.ui.status.innerText = 'WAKING CAMERA...';
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' }
            });
            this.video.srcObject = stream;
            await this.video.play();

            this.canvas.width = 640;
            this.canvas.height = 480;

            // 4. Bind UI
            this.setupUI();

            // 5. Hide Loader
            this.ui.overlay.style.opacity = '0';
            setTimeout(() => this.ui.overlay.style.display = 'none', 500);
            this.ui.status.innerText = 'STUDIO ONLINE';

            // 6. Start Loops
            this.renderLoop();
            this.aiLoop();

        } catch (err) {
            console.error("App Init Error:", err);
            this.ui.status.innerText = "CRITICAL ERROR: " + err.message;
        }
    }

    setupUI() {
        this.ui.toggleEarrings.onclick = () => {
            this.state.showEarrings = !this.state.showEarrings;
            this.ui.toggleEarrings.classList.toggle('active', this.state.showEarrings);
        };

        this.ui.toggleNecklace.onclick = () => {
            this.state.showNecklace = !this.state.showNecklace;
            this.ui.toggleNecklace.classList.toggle('active', this.state.showNecklace);
        };

        // Asset selection logic
        document.querySelectorAll('.asset-card').forEach(card => {
            card.onclick = async () => {
                const type = card.dataset.type;
                const path = card.dataset.path;
                
                document.querySelectorAll(`.asset-card[data-type="${type}"]`).forEach(c => c.classList.remove('active'));
                card.classList.add('active');

                if (type === 'earring') await this.earrings.loadAsset(path);
                if (type === 'necklace') await this.necklace.loadAsset(path);
            };
        });
    }

    async aiLoop() {
        if (this.state.isAIProcessing || this.video.readyState < 2) {
            return setTimeout(() => this.aiLoop(), 10);
        }

        this.state.isAIProcessing = true;
        const start = performance.now();

        try {
            // Run YOLO and Pose in parallel
            const [yoloDetections, poseData] = await Promise.all([
                this.yolo.detect(this.video),
                this.pose.detect(this.video)
            ]);

            this.state.lastYoloDetections = yoloDetections;
            this.state.lastPoseData = poseData;
            
            // Perform Fusion
            this.state.lastFusion = this.fusion.fuse(
                yoloDetections, 
                poseData, 
                this.canvas.width, 
                this.canvas.height
            );

            this.ui.fps.innerText = Math.round(performance.now() - start);
        } catch (err) {
            console.warn("AI Loop Error:", err);
        }

        this.state.isAIProcessing = false;
        setTimeout(() => this.aiLoop(), 5);
    }

    renderLoop() {
        if (this.video.readyState >= 2) {
            // 1. Draw Mirror Feed
            this.ctx.save();
            this.ctx.translate(this.canvas.width, 0);
            this.ctx.scale(-1, 1);
            this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
            this.ctx.restore();

            // 2. Render Jewellery
            if (this.state.showEarrings) {
                this.earrings.updateAndRender(this.state.lastYoloDetections);
            }

            if (this.state.showNecklace && this.state.lastFusion) {
                this.necklace.render(this.state.lastFusion);
            }
        }
        requestAnimationFrame(() => this.renderLoop());
    }
}

// Global initialization
const app = new GlimmerApp();
document.getElementById('start-btn').onclick = () => app.init();
