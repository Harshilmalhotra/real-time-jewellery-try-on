import { Spring } from '../physics/spring.js';

/**
 * 👂 Earring Engine Module
 * Handles tracking and physics-based rendering of earrings.
 */

class EarringInstance {
    constructor(x, y, size) {
        this.x = x;
        this.y = y;
        this.size = size;
        this.spring = new Spring(0.06, 0.93);
        this.angle = 0;
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
            // Swing effect based on horizontal movement
            this.spring.velocity -= dx * 0.18;
            this.lastX = this.x;
        } else {
            this.missedFrames++;
            if (this.missedFrames > 15) this.active = false;
        }

        this.spring.setTarget(0); // Aim for vertical rest
        this.angle = this.spring.update();
    }
}

export class EarringEngine {
    constructor(ctx) {
        this.ctx = ctx;
        this.instances = [];
        this.img = new Image();
    }

    async loadAsset(path) {
        return new Promise((res) => {
            this.img.onload = res;
            this.img.src = path;
        });
    }

    updateAndRender(detections) {
        if (!this.img.complete) return;

        let used = new Set();
        this.instances.forEach(t => {
            let bestIdx = -1;
            let bestDist = 80;
            detections.forEach((det, i) => {
                if (det.label !== 0) return; // Only earlobes (assuming class 0)
                let dist = Math.hypot(t.x - det.cx, t.y - det.cy);
                if (!used.has(i) && dist < bestDist) {
                    bestIdx = i; bestDist = dist;
                }
            });

            if (bestIdx !== -1) {
                let det = detections[bestIdx];
                t.update(det.cx, det.cy, det.w, true);
                used.add(bestIdx);
            } else {
                t.update(0, 0, 0, false);
            }
        });

        detections.forEach((det, i) => {
            if (!used.has(i) && det.label === 0) {
                this.instances.push(new EarringInstance(det.cx, det.cy, det.w));
            }
        });

        this.instances = this.instances.filter(t => t.active);

        this.instances.forEach(t => {
            if (t.missedFrames > 4) return;
            
            this.ctx.save();
            this.ctx.translate(t.x, t.y);
            this.ctx.rotate(t.angle * Math.PI / 180);
            
            let scale = (t.size * 1) / this.img.width;
            let ew = this.img.width * scale;
            let eh = this.img.height * scale;
            
            this.ctx.drawImage(this.img, -ew/2, 0, ew, eh);
            this.ctx.restore();
        });
    }
}
