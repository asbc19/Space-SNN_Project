# Space-SNN Project
- Hardware Accelerator for StereoSpike Network.

## 1. Design Space Exploration (DSE)
- Define the suitable hardware architecture to leverage all the StereoSpike's hardware-friendly characteristics.
  1. Binary inter-layer activations.
  2. Stateless PLIF neurons.
  3. Depthwise Separable (DWS) convolutions.
- A Jupiter Notebook is included with all the details.     

## 2. How to use the Jupiter Notebook
- Detailed information regarding the math formulas are included in the Notebook.

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

3. Memory traffic and latency can be analyzed for each conventional hardware architecture for DWS convolution.

https://github.com/user-attachments/assets/081f55ab-b986-46d3-97d0-a6ac80161c5e


3.1. Memory Traffic
  - User can define the bitwidth for weights, pfmaps, and ofmaps.
```python
# Bitwidths for weights, pfmap, and ofmap
n_bits = (8, 16, 16)
```
  - The function provides a comparative analysis (box plots) and individual results.

https://github.com/user-attachments/assets/d584598d-f062-46c5-a375-4bf7d12b8e22

3.2. Latency
  - User can define the the array size for each architecture.
```python
# (row, column) --> (i,j)

# UE
ue_array = (49, 128)

# SE
se_array_dw = (49, 64)
se_array_pw = (64, 128)

# RE
re_array = (64, 128)
dw_cols_inter = 64  # Number of columns for DW in RE for inter-channel
dw_cols_inter_prediction = 127  # Number of columns for DW in RE for inter-channel operation in prediction layers
dw_cols_intra = 120  # Number of columns for DW in RE for intra-channel
```
  - The function provides a comparative analysis (histograms and box plots) and individual results.

https://github.com/user-attachments/assets/c6d55796-e5b5-418b-88eb-32b0195c9d8b




   
