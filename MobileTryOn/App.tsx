import React from 'react';
import { StatusBar, View, Text } from 'react-native';
import { TryOnScreen } from './src/screens/TryOnScreen';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import { useCameraPermission } from 'react-native-vision-camera';

const App = () => {
  const { hasPermission, requestPermission } = useCameraPermission();

  React.useEffect(() => {
    console.log('[App] 🚀 Application Starting...');
    console.log('[App] 🔍 Checking Camera Permissions...');
    requestPermission().then((granted) => {
      console.log('[App] Permission Result:', granted);
    });
  }, [requestPermission]);

  return (
    <GestureHandlerRootView style={{ flex: 1 }}>
      <StatusBar barStyle="light-content" translucent backgroundColor="transparent" />
      {hasPermission ? (
        <TryOnScreen />
      ) : (
        <View style={{ flex: 1, backgroundColor: '#000', justifyContent: 'center', alignItems: 'center' }}>
          <Text style={{ color: '#FFF' }}>Waiting for Camera Permission...</Text>
        </View>
      )}
    </GestureHandlerRootView>
  );
};

export default App;
