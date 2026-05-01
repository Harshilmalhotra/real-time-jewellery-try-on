/*
 * Copyright (c) 2026 Harshil Malhotra. All rights reserved.
 * This code is subject to the terms of the Custom Non-Commercial & Attribution License 
 * found in the LICENSE.md file in the root directory of this source tree.
 * Commercial use requires a paid license.
 */

/**
 * 🌀 Spring Physics Module
 * Handles oscillatory motion for jewellery.
 */

export class Spring {
    constructor(stiffness, damping, mass = 1) {
        this.stiffness = stiffness;
        this.damping = damping;
        this.mass = mass;
        this.position = 0;
        this.velocity = 0;
        this.target = 0;
    }

    update() {
        const force = -this.stiffness * (this.position - this.target);
        const acceleration = force / this.mass;
        this.velocity = (this.velocity + acceleration) * this.damping;
        this.position += this.velocity;
        return this.position;
    }

    setTarget(v) {
        this.target = v;
    }
}
