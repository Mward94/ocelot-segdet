This directory contains train/val/test splits that were used to develop the tissue segmentation and cell detection models.

During model development, the GC training set was split into internal training and validation sets.

For the final models used to submit to the Grand Challenge, all data was moved to the training set and a final model trained.

A description of each of the splits are as follows:
* `cell_model`
  * Contains 357 of the 400 training images, with 321 in the training set, and 36 in the validation set
* `tissue_model`
  * Contains all 400 of the training images, with 364 in the training set, and 36 in the validation set (the same 36 as for the `cell_model` split)
* `cell_model_all_train`
  * Contains 357 images, all in the training set
* `tissue_model_all_train`
  * Contains 400 images, all in the training set
