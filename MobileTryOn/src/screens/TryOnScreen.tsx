import React from 'react';
import { StyleSheet, View, Text, Dimensions } from 'react-native';
import {
  Camera,
  useCameraDevice,
  useFrameOutput
} from 'react-native-vision-camera';
import { useSharedValue, useFrameCallback } from 'react-native-reanimated';
import { EarringOverlay } from '../components/EarringOverlay';
import { useTensorflowModel } from 'react-native-fast-tflite';

const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

// Default earlobe positions for a centered face in a selfie
const DEFAULT_LEFT_X  = SCREEN_WIDTH  * 0.14;
const DEFAULT_RIGHT_X = SCREEN_WIDTH  * 0.86;
const DEFAULT_EAR_Y   = SCREEN_HEIGHT * 0.42;
const DEFAULT_SIZE    = 60;

export const TryOnScreen = () => {
  const device = useCameraDevice('front');
  const plugin = useTensorflowModel(require('../assets/best.tflite'), []);

  const isInferring = useSharedValue(false);

  React.useEffect(() => {
    console.log('[TryOn] 📱 Screen Loaded');
  }, []);

  React.useEffect(() => {
    if (plugin.state === 'loaded') {
      console.log('[TryOn] ✅ YOLO Model Ready');
    } else if (plugin.state === 'error') {
      console.error('[TryOn] ❌ Model failed to load:', plugin.error);
    } else {
      console.log('[TryOn] 🧠 Loading YOLO Model...');
    }
  }, [plugin.state]);

  // Left earring
  const leftX    = useSharedValue(DEFAULT_LEFT_X);
  const leftY    = useSharedValue(DEFAULT_EAR_Y);
  const leftSize = useSharedValue(DEFAULT_SIZE);
  const leftAngle = useSharedValue(0);
  const leftVel   = useSharedValue(0);
  const lastLeftX = useSharedValue(DEFAULT_LEFT_X);

  // Right earring
  const rightX    = useSharedValue(DEFAULT_RIGHT_X);
  const rightY    = useSharedValue(DEFAULT_EAR_Y);
  const rightSize = useSharedValue(DEFAULT_SIZE);
  const rightAngle = useSharedValue(0);
  const rightVel   = useSharedValue(0);
  const lastRightX = useSharedValue(DEFAULT_RIGHT_X);

  const frameOutput = useFrameOutput({
    onFrame(frame) {
      'worklet';

      if (plugin.state !== 'loaded' || isInferring.value) {
        frame.dispose();
        return;
      }

      try {
        isInferring.value = true;

        const outputs = plugin.model.runSync([frame.getPixelBuffer()]);
        const data = new Float32Array(outputs[0]);

        // Find top-2 detections (left ear + right ear)
        let best1Conf = 0, best1Idx = -1;
        let best2Conf = 0, best2Idx = -1;

        for (let i = 0; i < 2100; i++) {
          const conf = data[4 * 2100 + i];
          if (conf > best1Conf) {
            best2Conf = best1Conf; best2Idx = best1Idx;
            best1Conf = conf;     best1Idx = i;
          } else if (conf > best2Conf) {
            best2Conf = conf; best2Idx = i;
          }
        }

        if (best1Conf > 0.35 && best1Idx !== -1) {
          const x1 = data[0 * 2100 + best1Idx];
          const y1 = data[1 * 2100 + best1Idx];
          const w1 = data[2 * 2100 + best1Idx];
          // Front camera: mirror X
          const sx1 = SCREEN_WIDTH - (x1 / 320) * SCREEN_WIDTH;
          const sy1 = (y1 / 320) * SCREEN_HEIGHT;
          const sz1 = (w1 / 320) * SCREEN_WIDTH;

          if (best2Conf > 0.35 && best2Idx !== -1) {
            const x2 = data[0 * 2100 + best2Idx];
            const y2 = data[1 * 2100 + best2Idx];
            const w2 = data[2 * 2100 + best2Idx];
            const sx2 = SCREEN_WIDTH - (x2 / 320) * SCREEN_WIDTH;
            const sy2 = (y2 / 320) * SCREEN_HEIGHT;
            const sz2 = (w2 / 320) * SCREEN_WIDTH;

            // Assign left/right by screen X
            if (sx1 < sx2) {
              leftX.value = sx1; leftY.value = sy1; leftSize.value = sz1;
              rightX.value = sx2; rightY.value = sy2; rightSize.value = sz2;
            } else {
              leftX.value = sx2; leftY.value = sy2; leftSize.value = sz2;
              rightX.value = sx1; rightY.value = sy1; rightSize.value = sz1;
            }
          } else {
            // Only one detection — mirror it for the other ear
            const midX = SCREEN_WIDTH / 2;
            leftX.value  = midX - Math.abs(sx1 - midX);
            rightX.value = midX + Math.abs(sx1 - midX);
            leftY.value  = sy1; rightY.value = sy1;
            leftSize.value = sz1; rightSize.value = sz1;
          }
        }
      } catch (_err) {
        // inference error — keep last known positions
      } finally {
        isInferring.value = false;
        frame.dispose();
      }
    }
  });

  // Pendulum physics at 60fps for both earrings
  useFrameCallback(() => {
    'worklet';
    const gravity = 0.85;
    const damping = 0.94;
    const stiffness = 0.05;

    const dxL = leftX.value - lastLeftX.value;
    leftVel.value += -dxL * 0.15;
    lastLeftX.value = leftX.value;
    const torqueL = -gravity * Math.sin(leftAngle.value * (Math.PI / 180));
    leftVel.value = (leftVel.value + torqueL - leftAngle.value * stiffness) * damping;
    leftAngle.value += leftVel.value;
    if (Math.abs(leftAngle.value) > 45) {
      leftAngle.value = Math.sign(leftAngle.value) * 45;
      leftVel.value *= -0.5;
    }

    const dxR = rightX.value - lastRightX.value;
    rightVel.value += -dxR * 0.15;
    lastRightX.value = rightX.value;
    const torqueR = -gravity * Math.sin(rightAngle.value * (Math.PI / 180));
    rightVel.value = (rightVel.value + torqueR - rightAngle.value * stiffness) * damping;
    rightAngle.value += rightVel.value;
    if (Math.abs(rightAngle.value) > 45) {
      rightAngle.value = Math.sign(rightAngle.value) * 45;
      rightVel.value *= -0.5;
    }
  });

  if (!device) {
    return (
      <View style={styles.container}>
        <Text style={styles.text}>Initializing Camera...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        outputs={[frameOutput]}
        onStarted={() => console.log('[TryOn] ✅ Camera Session Started')}
        onError={(e) => console.error('[TryOn] ❌ Camera Error:', e)}
      />

      <EarringOverlay
        left={{ x: leftX, y: leftY, size: leftSize, angle: leftAngle }}
        right={{ x: rightX, y: rightY, size: rightSize, angle: rightAngle }}
      />

      <View style={styles.header}>
        <Text style={styles.title}>GLIMMER AR TRY-ON</Text>
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#000' },
  header: {
    position: 'absolute',
    top: 50,
    width: '100%',
    alignItems: 'center',
  },
  title: {
    color: '#FFF',
    fontSize: 18,
    fontWeight: 'bold',
    letterSpacing: 2,
    textShadowColor: 'rgba(0,0,0,0.75)',
    textShadowOffset: { width: 0, height: 2 },
    textShadowRadius: 10,
  },
  text: { color: '#FFF' },
});
