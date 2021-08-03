# PPG-QI
PPG Quality Index (QI) models

This repo contains code for generating image/array pairs for waveform windows, and then training CNN models to predict whether or not a window is valid/invalid. 
The probability of a window being valid can be interpreted as a quality index.  

3 steps are required for creating the quality index models: (1) generating waveform images to label, (2) labeling the waveform images as valid/invalid, and (3) training a CNN model to generate the quality index. 

For additional details, please see [Imputation of the continuous arterial line blood pressure waveform from non-invasive measurements using deep learning](https://www.nature.com/articles/s41598-021-94913-y).

## Generating input data

## Loading data into Label Studio

1. Clone [Label Studio](https://labelstud.io) from the [Github repo](https://github.com/heartexlabs/label-studio)
2. Change directories into the cloned label-studio directory and install dependencies 
```
# Install all package dependencies
pip install -e .
```
3. Create a soft link to the directory containing the image files, so they can be served locally (see [this issue](https://github.com/heartexlabs/label-studio/issues/49) for details)
```
ln -s /path/to/ppg_images label-studio/static/ppg_images
```
4. Initialize the label-studio project, supplying the path to save the label-studio project files
```
python label_studio/server.py init ../wave_qi/wave_qi
```
5. Start the label-studio server, supplying the path to save label-studio project files
```
python label_studio/server.py start ../wave_qi/wave_qi
```
6. Click the "Setup" tab to configure the project settings

Set up the project as an "Image classification" ask, and set the Labeling Config like this: 
```
<View>
  <Image name="image" value="$image"/>
  <Choices name="ppg" toName="image" showInLine="true">
    <Choice value="invalid" background="red"/>
    <Choice value="neutral" background="blue" />
    <Choice value="valid" background="green" />
  </Choices>
</View>
```
And click Save at the bottom of the page. 
This will give you three classification options per image: (1) invalid, (2) neutral (not sure), and (3) valid. 

7. To to the Import tab to load the `[signal]_image_file_paths.csv` file that contains locations of the waveform images for labeling
8. Go to the Labeling tab and start labeling images. Note that the number keys can be used as keyboard shortcuts for labels. 
9. Once windows are labeled, go to the Export tab and choose CSV as the export format to save the labeling results as a .csv file

## Training the QI models

## Citation

If you use the PPG QI model in your work, please cite:

```
Hill, B.L., Rakocz, N., Rudas, Á. et al. Imputation of the continuous arterial line 
blood pressure waveform from non-invasive measurements using deep learning. 
Sci Rep 11, 15755 (2021). https://doi.org/10.1038/s41598-021-94913-y
```
