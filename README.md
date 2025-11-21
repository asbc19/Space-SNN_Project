# Space-SNN Project
- Hardware Accelerator for StereoSpike Network.

## 1. Design Space Exploration (DSE)
- Define the suitable hardware architecture to leverage all the StereoSpike's hardware-friendly characteristics.
  1. Binary inter-layer activations.
  2. Stateless PLIF neurons.
  3. Depthwise Separable (DWS) convolutions.
- A Jupiter Notebook is included with all the details.     

## 2. How to use the Jupiter Notebook
1. Import libraries and customized functions.

https://github.com/user-attachments/assets/357162b3-1adc-40a1-a9e9-779b417108e3

2. User can input the network information: filter size and dimensions of individual layers.
```python
# Stereospike DW Filter (R, S)
# All layers have the same DW filter size
filter_size = (7, 7)

network_layers: List[ConvLayerSpec] = [
    ConvLayerSpec("stem", (4, 260, 346), (32, 260, 346)),
    ConvLayerSpec("conv1", (32, 260, 346), (64, 130, 173)),
    ConvLayerSpec("conv2", (64, 130, 173), (128, 65, 87)),
    ConvLayerSpec("conv3", (128, 65, 87), (256, 33, 44)),
    ConvLayerSpec("conv4", (256, 33, 44), (512, 17, 22)),
    ConvLayerSpec("bottleneck", (512, 17, 22), (512, 17, 22)),
    ConvLayerSpec("deconv4", (512, 17, 22), (256, 33, 44)),
    ConvLayerSpec("deconv3", (256, 33, 44), (128, 65, 87)),
    ConvLayerSpec("deconv2", (128, 65, 87), (64, 130, 173)),
    ConvLayerSpec("deconv1", (64, 130, 173), (32, 260, 346)),
    ConvLayerSpec("depth4", (256, 33, 44), (1, 260, 346)),
    ConvLayerSpec("depth3", (128, 65, 87), (1, 260, 346)),
    ConvLayerSpec("depth2", (64, 130, 173), (1, 260, 346)),
    ConvLayerSpec("depth1", (32, 260, 346), (1, 260, 346))]
```
