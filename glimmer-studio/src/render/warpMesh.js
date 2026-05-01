/*
 * Copyright (c) 2026 Harshil Malhotra. All rights reserved.
 * This code is subject to the terms of the Custom Non-Commercial & Attribution License 
 * found in the LICENSE.md file in the root directory of this source tree.
 * Commercial use requires a paid license.
 */

/**
 * 🕸️ Warp Mesh Module
 * Provides vertex-based image deformation for realistic jewellery placement.
 */

export class WarpMesh {
    constructor(ctx) {
        this.ctx = ctx;
    }

    /**
     * Draws an image warped by a grid of points
     * @param {HTMLImageElement} img 
     * @param {Array} points - Array of {x, y} for the grid
     * @param {number} cols - Number of columns in grid
     * @param {number} rows - Number of rows in grid
     */
    drawWarpedImage(img, points, cols, rows) {
        const iw = img.width;
        const ih = img.height;

        for (let r = 0; r < rows - 1; r++) {
            for (let c = 0; c < cols - 1; c++) {
                const p1 = points[r * cols + c];
                const p2 = points[r * cols + c + 1];
                const p3 = points[(r + 1) * cols + c];
                const p4 = points[(r + 1) * cols + c + 1];

                const u1 = (c / (cols - 1)) * iw;
                const v1 = (r / (rows - 1)) * ih;
                const u2 = ((c + 1) / (cols - 1)) * iw;
                const v2 = ((r + 1) / (rows - 1)) * ih;

                this.drawTriangle(img, p1, p2, p3, u1, v1, u2, v1, u1, v2);
                this.drawTriangle(img, p2, p4, p3, u2, v1, u2, v2, u1, v2);
            }
        }
    }

    drawTriangle(img, p1, p2, p3, u1, v1, u2, v2, u3, v3) {
        const ctx = this.ctx;
        ctx.save();
        ctx.beginPath();
        ctx.moveTo(p1.x, p1.y);
        ctx.lineTo(p2.x, p2.y);
        ctx.lineTo(p3.x, p3.y);
        ctx.closePath();
        ctx.clip();

        // Calculate affine transform matrix
        const det = (u1 - u3) * (v2 - v3) - (u2 - u3) * (v1 - v3);
        const a = ((p1.x - p3.x) * (v2 - v3) - (p2.x - p3.x) * (v1 - v3)) / det;
        const b = ((p2.x - p3.x) * (u1 - u3) - (p1.x - p3.x) * (u2 - u3)) / det;
        const c = p3.x - a * u3 - b * v3;
        const d = ((p1.y - p3.y) * (v2 - v3) - (p2.y - p3.y) * (v1 - v3)) / det;
        const e = ((p2.y - p3.y) * (u1 - u3) - (p1.y - p3.y) * (u2 - u3)) / det;
        const f = p3.y - d * u3 - e * v3;

        ctx.transform(a, d, b, e, c, f);
        ctx.drawImage(img, 0, 0);
        ctx.restore();
    }
}
