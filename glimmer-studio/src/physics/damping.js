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
