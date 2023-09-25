# OCELOT Cell Detection Leveraging Tissue Segmentation
This repository contains code that can be used to train and evaluate models that were submitted for the [OCELOT Grand Challenge](https://ocelot2023.grand-challenge.org/ocelot2023/)

![](./img/algorithm.jpg)

This also includes files provided by the [OCELOT Grand Challenge](https://github.com/lunit-io/ocelot23algo) for:
* Testing, Building, and Exporting the Docker image
* Running evaluation on generated predictions
* Running the inference tool on image data

The structure of this repository generally follows the structure provided by the [OCELOT Grand Challenge](https://github.com/lunit-io/ocelot23algo).

The training and inference parts of the repository are configured to match what was used in the OCELOT Grand Challenge.

## Running Evaluation on Generated Predictions
Below are instructions on running evaluation on predictions generated by the `process.py` script: 

1. Follow the instructions in `evaluation/README.md` to convert annotation CSV files to format required by evaluation code
2. Create a directory that contains the images you want to evaluate on (in the same structure as the existing test images).
   1. Make a copy of the metadata file, and leave in the images you're evaluating over. NOTE: See the sample metadata format. Ordering is based on indexing into the list.
   2. It's very important you get the ordering right. On the image loading side, it's based on ordering of the filenames. In the metadata file, it's based on how you order the list.
   3. It's not ideal, but it's based on how it was already implemented. Also based on how the convert_gt_csvs_to_json script works
3. Modify `util/constants.py` so the filepaths point to your images.
4. Place your model weights in `user/weights`. You can follow the instructions in `user/weights/README.md` to download the set of pretrained weights used for the OCELOT Grand Challenge submission.
5. Run `python process.py`
6. Move the generated predictions to the expected location in `evaluation/eval.py`
7. Run `python evaluation/eval.py`
