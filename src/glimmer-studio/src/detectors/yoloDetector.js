/*
 * Copyright (c) 2026 Harshil Malhotra. All rights reserved.
 * This code is subject to the terms of the Custom Non-Commercial & Attribution License 
 * found in the LICENSE.md file in the root directory of this source tree.
 * Commercial use requires a paid license.
 */

/**
 * 🎯 YOLOv8 ONNX Detector Module
 * Handles high-precision detection of earlobes, eyes, and nose.
 */

export class YoloDetector {
    constructor(config) {
        this.config = config;
        this.session = null;
        this.canvas = new OffscreenCanvas(320, 320);
        this.ctx = this.canvas.getContext('2d', { alpha: false });
    }

    async init() {
        try {
            // Load ONNX Runtime session
            this.session = await ort.InferenceSession.create(this.config.modelPath, {
                executionProviders: ['wasm'],
                graphOptimizationLevel: 'all'
            });
            return true;
        } catch (err) {
            console.error("YOLO Init Error:", err);
            return false;
        }
    }

    async detect(video) {
        if (!this.session) return [];

        const cropSize = 384; 
        const offsetX = (video.videoWidth - cropSize) / 2;
        const offsetY = (video.videoHeight - cropSize) / 2;
        
        this.ctx.save();
        this.ctx.scale(-1, 1);
        this.ctx.drawImage(video, offsetX, offsetY, cropSize, cropSize, -320, 0, 320, 320);
        this.ctx.restore();
        
        const imgData = this.ctx.getImageData(0, 0, 320, 320).data;
        const input = new Float32Array(3 * 320 * 320);
        const area = 320 * 320;
        
        for (let i = 0; i < area; i++) {
            const i4 = i * 4;
            input[i] = imgData[i4] / 255.0;
            input[i + area] = imgData[i4 + 1] / 255.0;
            input[i + area * 2] = imgData[i4 + 2] / 255.0;
        }

        const tensor = new ort.Tensor('float32', input, [1, 3, 320, 320]);
        const results = await this.session.run({ images: tensor });
        const output = results[Object.keys(results)[0]].data;
        
        const detections = [];
        const boxes = 2100;
        const scaleFactor = cropSize / 320; 
        
        for (let i = 0; i < boxes; i++) {
            let maxConf = 0, clsIdx = -1;
            for (let j = 0; j < 4; j++) {
                let conf = output[(4 + j) * boxes + i];
                if (conf > maxConf) { maxConf = conf; clsIdx = j; }
            }

            if (maxConf > this.config.confThreshold) {
                detections.push({
                    cx: (output[i] * scaleFactor) + offsetX, 
                    cy: (output[boxes + i] * scaleFactor) + offsetY,
                    w: output[2 * boxes + i] * scaleFactor, 
                    label: clsIdx,
                    confidence: maxConf
                });
            }
        }
        return detections;
    }
}
