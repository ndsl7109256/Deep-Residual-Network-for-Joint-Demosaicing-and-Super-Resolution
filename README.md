# Deep-Residual-Network-for-Joint-Demosaicing-and-Super-Resolution
implement the paper 'Deep Residual Network for Joint Demosaicing and Super-Resolution' with Keras
# Deep-Residual-Network-for-Joint-Demosaicing-and-Super-Resolution
implement the paper 'Deep Residual Network for Joint Demosaicing and Super-Resolution' with Keras


## Step 1 prepare data

use `prepare.py` to split images to 64x64x1 and 128x128x3 to get the X and Y


## Step 2 train model
with the train set, we can use `train.py` to train our model

## Step 3 prepare test
For validating,we just need to use `prepare_test.py` resize images and get the bayer value.
## Step 4 Validating
Use `validate.py` to predict image and caculate CSPNR. 
