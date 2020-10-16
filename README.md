# Wave-QI
Waveform Quality Index (QI) models

This repo contains code for generating image/array pairs for waveform windows, and then training CNN models to predict whether or not a window is valid/invalid. 
The probability of a window being valid can be interpreted as a quality index.  

3 steps are required for creating the quality index models: (1) generating waveform images to label, (2) labeling the waveform images as valid/invalid, and (3) training a CNN model to generate the quality index. 

## Generating input data

## Loading data into Label Studio

1. Clone [Label Studio](https://labelstud.io) from the [Github repo](https://github.com/heartexlabs/label-studio)
2. Change directories into the cloned label-studio directory and install dependencies 
```
# Install all package dependencies
pip install -e .
```
3. Start the label-studio server, supplying the path to save label-studio project files
```
python label_studio/server.py start ../wave_qi/wave_qi
```
4. Load the .csv file that contains locations of the waveform images for labeling
5. Once windows are labeled, export the labeling results as a .csv file

## Training the QI models

