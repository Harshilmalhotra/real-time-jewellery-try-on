/*
 * Copyright (c) 2026 Harshil Malhotra. All rights reserved.
 * This code is subject to the terms of the Custom Non-Commercial & Attribution License 
 * found in the LICENSE.md file in the root directory of this source tree.
 * Commercial use requires a paid license.
 */

/**
 * 🌊 Damping Physics Module
 * Smooths out transitions and handles inertia.
 */

export class Damping {
    constructor(factor) {
        this.factor = factor;
        this.value = 0;
    }

    apply(current, target) {
        this.value = this.value * this.factor + (target - current) * (1 - this.factor);
        return current + this.value;
    }

    static lerp(start, end, amt) {
        return (1 - amt) * start + amt * end;
    }
}
