# Shape Reconstruction from Tactile images
Here are three ways to reconstruct the height maps from the tactile images

## Generate data
- `all_test.py` includes the running all three methods on ycb dataset, you can set the running flag to True to run either one or all; and pick one object to generate data or all objects under the tactile data folders
- `config.py` includes the configurations of all methods, the path to data and calibration files
- `lookup.py`, `mlp.py`, `fcrn.py` and `contact_mask.py` are the scripts to run each of them.

## Lookup Table
One-to-one corresponding from image intensity to normals.
This method doesn't handle the spacial variance.
- `lookupCalibrate.py` is to calibrate the lookup table.
- `lookupReconstruct.py` is to reconstruct the height maps from lookup table.

## MLP Net
This way approximates the mapping f(r,g,b,u,v) = n. u & v is the location on the image plane.
- `netCalibrate.py` is to generate data pack for learning.
- `netReconstruct.py` is to reconstruct the height maps from the trained net.

## FCRN
End-to-end training on tactile images and height maps
- `/FCRN/test_dataset.py` is used to input the tactile images and output the heightmaps
- the trained model is on Google Drive under the calib called `checkpoint.pth.tar`
