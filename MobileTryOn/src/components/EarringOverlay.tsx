import React from 'react';
import { StyleSheet } from 'react-native';
import { Canvas, Image, useImage, Group } from '@shopify/react-native-skia';
import { useDerivedValue, SharedValue } from 'react-native-reanimated';

interface EarValues {
  x: SharedValue<number>;
  y: SharedValue<number>;
  size: SharedValue<number>;
  angle: SharedValue<number>;
}

interface Props {
  left: EarValues;
  right: EarValues;
}

const TARGET_WIDTH = 60;

function EarringGroup({ ear, image, imgHeight }: { ear: EarValues; image: any; imgHeight: number }) {
  const transform = useDerivedValue(() => [
    { translateX: ear.x.value },
    { translateY: ear.y.value },
    { rotate: (ear.angle.value * Math.PI) / 180 },
  ]);
  const origin = useDerivedValue(() => ({ x: ear.x.value, y: ear.y.value }));

  return (
    <Group origin={origin} transform={transform}>
      <Image
        image={image}
        x={-(TARGET_WIDTH / 2)}
        y={0}
        width={TARGET_WIDTH}
        height={imgHeight}
      />
    </Group>
  );
}

export const EarringOverlay: React.FC<Props> = ({ left, right }) => {
  const image = useImage(require('../assets/earring.png'));

  if (!image) return null;

  const imgWidth = (typeof image.width === 'function' ? image.width() : image.width) as number;
  const imgHeight = (typeof image.height === 'function' ? image.height() : image.height) as number;

  if (!imgWidth || !imgHeight) return null;

  const scaledHeight = imgHeight * (TARGET_WIDTH / imgWidth);

  return (
    <Canvas style={StyleSheet.absoluteFill} pointerEvents="none">
      <EarringGroup ear={left}  image={image} imgHeight={scaledHeight} />
      <EarringGroup ear={right} image={image} imgHeight={scaledHeight} />
    </Canvas>
  );
};
