/*
 * Copyright (c) 2026 Harshil Malhotra. All rights reserved.
 * This code is subject to the terms of the Custom Non-Commercial & Attribution License 
 * found in the LICENSE.md file in the root directory of this source tree.
 * Commercial use requires a paid license.
 */

/**
 * ✨ Glimmer AR Engine v3.0 (MODULAR)
 * Integrated YOLOv8 + MediaPipe Pose Fusion
 */

import { YoloDetector } from './src/detectors/yoloDetector.js';
import { PoseDetector } from './src/detectors/poseDetector.js';
import { LandmarkFusion } from './src/fusion/landmarkFusion.js';
import { EarringEngine } from './src/jewellery/earringEngine.js';
import { NecklaceRenderer } from './src/render/necklaceRenderer.js';
import { JEWELLERY_ASSETS } from './jewellery/manifest.js';

const CONFIG = {
    yolo: {
        modelPath: 'best.onnx',
        confThreshold: 0.45
    }
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
            toggleEarrings: document.getElementById('toggle-earrings'),
            toggleNecklace: document.getElementById('toggle-necklace'),
            earringGrid: document.getElementById('earring-grid'),
            necklaceGrid: document.getElementById('necklace-grid')
        };

        // Populate UI immediately
        this.populateAssetGrids();
        this.setupUI();
    }

    async init(addLog) {
        try {
            this.ui.status.innerText = 'INITIALIZING NEURAL CORES...';
            if (addLog) await addLog("Initializing Neural Cores...", 500);
            
            // 1. Initialize Detectors
            await Promise.all([
                this.yolo.init(),
                this.pose.init()
            ]);

            if (addLog) await addLog("Neural models loaded successfully.", 500);

            // 2. Load Default Assets
            const defaultEarring = JEWELLERY_ASSETS.earrings[0].path;
            const defaultNecklace = JEWELLERY_ASSETS.necklaces[0].path;

            if (addLog) await addLog("Syncing digital jewellery assets...", 400);

            await Promise.all([
                this.earrings.loadAsset(defaultEarring),
                this.necklace.loadAsset(defaultNecklace)
            ]);

            // 3. Setup Webcam
            this.ui.status.innerText = 'WAKING CAMERA...';
            if (addLog) await addLog("Requesting camera permissions...", 500);
            
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, facingMode: 'user' }
            });
            
            if (addLog) await addLog("Camera connected. Aligning quantum mirrors...", 600);
            
            this.video.srcObject = stream;
            await this.video.play();

            this.canvas.width = 640;
            this.canvas.height = 480;

            if (addLog) await addLog("Calibration complete. Entering Studio.", 600);

            // 5. Hide Loader
            this.ui.overlay.style.opacity = '0';
            this.ui.overlay.style.backdropFilter = 'blur(0px)';
            setTimeout(() => this.ui.overlay.style.display = 'none', 1000);
            this.ui.status.innerText = 'STUDIO ONLINE';

            // 6. Start Loops
            this.renderLoop();
            this.aiLoop();

        } catch (err) {
            console.error("App Init Error:", err);
            this.ui.status.innerText = "CRITICAL ERROR: " + err.message;
            if (addLog) await addLog(`[CRITICAL FAULT] ${err.message}`, 100);
        }
    }

    populateAssetGrids() {
        const createCard = (asset, type) => {
            const card = document.createElement('div');
            card.className = 'asset-card';
            card.dataset.type = type;
            card.dataset.path = asset.path;
            card.innerHTML = `<img src="${asset.path}" alt="${asset.name}">`;
            
            // Mark first item as active
            if (asset.path === JEWELLERY_ASSETS[type === 'earring' ? 'earrings' : 'necklaces'][0].path) {
                card.classList.add('active');
            }
            
            return card;
        };

        JEWELLERY_ASSETS.earrings.forEach(asset => {
            this.ui.earringGrid.appendChild(createCard(asset, 'earring'));
        });

        JEWELLERY_ASSETS.necklaces.forEach(asset => {
            this.ui.necklaceGrid.appendChild(createCard(asset, 'necklace'));
        });
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

        // Delegate asset selection logic to the grids
        const handleCardClick = async (e) => {
            const card = e.target.closest('.asset-card');
            if (!card) return;

            const type = card.dataset.type;
            const path = card.dataset.path;
            
            document.querySelectorAll(`.asset-card[data-type="${type}"]`).forEach(c => c.classList.remove('active'));
            card.classList.add('active');

            if (type === 'earring') await this.earrings.loadAsset(path);
            if (type === 'necklace') await this.necklace.loadAsset(path);
        };

        this.ui.earringGrid.onclick = handleCardClick;
        this.ui.necklaceGrid.onclick = handleCardClick;
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

document.getElementById('turn-on-mirror-btn').onclick = async () => {
    const btn = document.getElementById('turn-on-mirror-btn');
    const logs = document.getElementById('techy-logs');
    const logContainer = document.getElementById('log-container');
    const magicCircle = document.getElementById('magic-circle');
    
    btn.style.display = 'none';
    logs.style.display = 'flex';
    magicCircle.style.display = 'block';

    const addLog = (msg, delay) => new Promise(resolve => {
        setTimeout(() => {
            const div = document.createElement('div');
            div.className = 'log-line';
            div.innerText = `> ${msg}`;
            logContainer.appendChild(div);
            // Auto-scroll logic if logs get too long
            if (logContainer.children.length > 4) {
                logContainer.removeChild(logContainer.firstChild);
            }
            resolve();
        }, delay);
    });

    await addLog("Magical mirror is unfolding...", 100);
    await addLog("Verifying neural pathways...", 800);
    
    app.init(addLog);
};
