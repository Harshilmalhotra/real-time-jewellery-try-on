export class EarringPhysics {
  x: number;
  y: number;
  size: number;
  angle = 0;
  vel = 0;
  lastX: number;

  constructor(x: number, y: number, size: number) {
    this.x = x;
    this.y = y;
    this.size = size;
    this.lastX = x;
  }

  /**
   * Updates the physics state based on new detection coordinates.
   * @param tx Target X (from detector)
   * @param ty Target Y (from detector)
   * @param ts Target Size (from detector)
   * @param detected Whether a detection was made in the current frame
   */
  update(tx: number, ty: number, ts: number, detected = true) {
    if (detected) {
      // Smoothing (Lerp) to reduce detection noise
      this.x = this.x * 0.3 + tx * 0.7;
      this.y = this.y * 0.3 + ty * 0.7;
      this.size = this.size * 0.7 + ts * 0.3;

      // Calculate horizontal velocity for the swing effect
      const dx = this.x - this.lastX;
      this.vel += -dx * 0.15; // SWING_SENSITIVITY
      this.lastX = this.x;
    }

    // Pendulum Physics Constants
    const gravity = 0.85;
    const damping = 0.94;
    const restoringStiffness = 0.05;

    // Physics Simulation
    const torque = -gravity * Math.sin(this.angle * (Math.PI / 180));
    const restoring = -this.angle * restoringStiffness;

    this.vel = (this.vel + torque + restoring) * damping;
    this.angle += this.vel;

    // Clamp angle to prevent unnatural 360 loops
    if (Math.abs(this.angle) > 45) {
      this.angle = Math.sign(this.angle) * 45;
      this.vel *= -0.5; // Bounce back
    }
  }
}
