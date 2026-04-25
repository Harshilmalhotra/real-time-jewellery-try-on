/**
 * 🔗 Landmark Fusion Module
 * Merges YOLO detections and Pose landmarks into a stable coordinate system.
 */

export class LandmarkFusion {
    constructor() {
        this.history = [];
        this.maxHistory = 5;
    }

    fuse(yoloDetections, poseData, videoWidth, videoHeight) {
        if (!poseData) return null;

        const nose = {
            x: poseData.nose.x * videoWidth,
            y: poseData.nose.y * videoHeight
        };

        // Convert normalized pose coordinates to pixel coordinates
        const leftShoulder = {
            x: poseData.leftShoulder.x * videoWidth,
            y: poseData.leftShoulder.y * videoHeight
        };
        const rightShoulder = {
            x: poseData.rightShoulder.x * videoWidth,
            y: poseData.rightShoulder.y * videoHeight
        };

        const shoulderDist = Math.hypot(leftShoulder.x - rightShoulder.x, leftShoulder.y - rightShoulder.y);
        const avgShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
        
        // COLLAR BONE HEIGHT: Approximately 67% of the way from nose to shoulders
        // If nose is missing or pose is weird, fallback to shoulder line - 25%
        const clavicleY = nose ? (nose.y + (avgShoulderY - nose.y) * 0.67) : (avgShoulderY - shoulderDist * 0.25);

        const inwardOffset = 0.30; // 30% inward from shoulders

        // Compute Clavicle Anchors
        const leftAnchor = {
            x: leftShoulder.x + (rightShoulder.x - leftShoulder.x) * inwardOffset,
            y: clavicleY
        };

        const rightAnchor = {
            x: rightShoulder.x + (leftShoulder.x - rightShoulder.x) * inwardOffset,
            y: clavicleY
        };

        // Necklace center
        const neckCenter = {
            x: (leftAnchor.x + rightAnchor.x) / 2,
            y: (leftAnchor.y + rightAnchor.y) / 2
        };

        // Width based on clavicle distance
        const necklaceWidth = Math.hypot(leftAnchor.x - rightAnchor.x, leftAnchor.y - rightAnchor.y) * 1.4;

        // Rotation angle
        const angle = Math.atan2(rightAnchor.y - leftAnchor.y, rightAnchor.x - leftAnchor.x);

        // Torso tilt
        const tilt = (nose.x - neckCenter.x) / shoulderDist;

        return {
            neckCenter,
            leftAnchor,
            rightAnchor,
            width: necklaceWidth,
            angle,
            tilt,
            shoulderDist
        };
    }
}
