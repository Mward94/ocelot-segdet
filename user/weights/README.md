# Weights Directory

You can use this directory to store model weights files that are used by the `inference.py` script.

## Pre-Trained Weights
To use the same weights that were used in the OCELOT submission:

1. Download the tissue model weights from [here](https://drive.google.com/file/d/1gPHIJVLdSO29eP4p2p_S8VFDznNAHgxi/view?usp=sharing)
2. Place them in this directory with the filename `tissue_model_weights.pth`
3. Download the cell model weights from [here](https://drive.google.com/file/d/1kOUgCgJpXqpxqrMwckJ92kFMJU22IZi0/view?usp=sharing)
4. Place them in this directory with the filename `cell_model_weights.pth`

Then you can use the `process.py` script to run the algorithm in the same way it was run in the OCELOT Grand Challenge.

Brief description about how training of pretrained models

**Tissue Model**
* SegFormer-B0
* 1500 Epochs
* 0.5 MPP
* Images Macenko normalised
* Trained on all 400 images

**Cell Model**
* SegFormer-B2
* 250 Epochs
* 0.2 MPP
* Images **NOT** Macenko normalised
* Trained on refined set of 357 images
