/**
 * 🧘 MediaPipe Pose Detector Module
 * Extracts shoulder and neck landmarks for necklace anchoring.
 */

export class PoseDetector {
    constructor() {
        this.pose = null;
        this.results = null;
    }

    async init() {
        // MediaPipe Pose is typically loaded via script tag, but we initialize it here
        this.pose = new Pose({
            locateFile: (file) => {
                return `https://cdn.jsdelivr.net/npm/@mediapipe/pose/${file}`;
            }
        });

        this.pose.setOptions({
            modelComplexity: 1,
            smoothLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        this.pose.onResults((results) => {
            this.results = results;
        });

        return true;
    }

    async detect(video) {
        await this.pose.send({ image: video });
        
        if (!this.results || !this.results.poseLandmarks) return null;

        const lm = this.results.poseLandmarks;
        
        // Extract specific landmarks (MediaPipe Pose Indices)
        // 0: nose, 11: left shoulder, 12: right shoulder, 7: left ear, 8: right ear
        return {
            nose: lm[0],
            leftShoulder: lm[11],
            rightShoulder: lm[12],
            leftEar: lm[7],
            rightEar: lm[8],
            all: lm
        };
    }
}
