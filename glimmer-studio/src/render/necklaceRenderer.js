import { WarpMesh } from './warpMesh.js';
import { Spring } from '../physics/spring.js';

/**
 * 📿 Necklace Renderer Module
 * Handles physics-based rendering of necklaces with curve deformation.
 */

export class NecklaceRenderer {
    constructor(ctx) {
        this.ctx = ctx;
        this.warper = new WarpMesh(ctx);
        this.img = new Image();
        this.pendantSpring = new Spring(0.05, 0.92); // stiffness, damping
        this.pendantX = 0;
    }

    async loadAsset(path) {
        return new Promise((res) => {
            this.img.onload = res;
            this.img.src = path;
        });
    }

    render(fusionData) {
        if (!fusionData || !this.img.complete) return;

        const { neckCenter, leftAnchor, rightAnchor, width, angle, tilt, shoulderDist } = fusionData;

        // Update physics for pendant swing
        this.pendantSpring.setTarget(neckCenter.x);
        this.pendantX = this.pendantSpring.update();

        // Generate a grid of points for the necklace curve
        const cols = 5;
        const rows = 3;
        const points = [];
        
        const h = width * (this.img.height / this.img.width);
        
        // Clavicle to Pendant Drop (16-18% of shoulder width)
        const dropDepth = shoulderDist * 0.17; 

        for (let r = 0; r < rows; r++) {
            for (let c = 0; c < cols; c++) {
                const t = c / (cols - 1); // 0 to 1
                const v = r / (rows - 1); // 0 to 1

                // Quadratic Bezier curve for the necklace hang
                // Control point for the Bezier curve is centered horizontally and dropped vertically
                const cpX = this.pendantX;
                const cpY = neckCenter.y + dropDepth;

                // Simple parabolic interpolation for the chain curve
                const x = this.lerp(leftAnchor.x, rightAnchor.x, t);
                const curveY = this.calculateCurve(leftAnchor.y, cpY, rightAnchor.y, t);
                
                // Add vertical offset for the row
                const y = curveY + (v * h * 0.5);

                // Add depth/tilt perspective
                const zOffset = (t - 0.5) * tilt * width * 0.3;
                
                points.push({ x: x + zOffset, y: y });
            }
        }

        this.ctx.globalAlpha = 0.95;
        this.warper.drawWarpedImage(this.img, points, cols, rows);
        this.ctx.globalAlpha = 1.0;
    }

    lerp(a, b, t) {
        return a + (b - a) * t;
    }

    calculateCurve(y1, y2, y3, t) {
        // Quadratic Bezier
        return (1 - t) * (1 - t) * y1 + 2 * (1 - t) * t * y2 + t * t * y3;
    }
}
