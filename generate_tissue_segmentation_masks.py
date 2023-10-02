"""Runs tissue segmentation on tissue images, then converts them to ones usable for cell model.

Inference run on 'tissue' images, then relevant areas cropped and extracted for 'cell' model (at
cell image MPP)
"""
import argparse
import os

import cv2
import numpy as np
import torch
from rasterio.enums import Resampling
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.tissue_dataset import TissueDataset
from networks.postprocessors.seg_softmax import SegSoftmax
from networks.segformer import SegFormer
from util.constants import TISSUE_CLASSES, SEG_MASK_PROB_KEY, TISSUE_CLASS_COLOURS
from util.gcio import read_json
from util.helpers import create_directory
from util.image import (
    crop_image, DEFAULT_TIF_PROFILE, write_tif_rasterio, load_tif_rasterio, overlay_images,
    write_image)
from util.ocelot_parsing import cell_scale_crop_in_tissue_at_cell_mpp
from util.torch import get_default_device, move_data_to_device
from util.training import postprocess_outputs

# Data related
DEFAULT_SPLIT_DIRECTORY = './splits/tissue_model' # _all_train'
DEFAULT_MPP = 0.5       # MPP to scale data to

# Inference related
DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 4
DEFAULT_OCM = 64

# Model related
DEFAULT_SEGFORMER_SIZE = 'b0'

# Expected size of cell image mask
OUTPUT_MASK_SHAPE = 1024


def parse_args():
    parser = argparse.ArgumentParser()

    # Data source
    parser.add_argument('--ocelot-directory', type=str,
                        help='Path to OCELOT data directory. Should contain metadata.json file.',
                        required=True)
    parser.add_argument('--data-directory', type=str,
                        help='Directory where processed data is stored', required=True)
    parser.add_argument('--split-directory', type=str,
                        help='Path to split directory. Masks will be generated for all images in '
                             'both the \'train\' and \'val\' splits',
                        default=DEFAULT_SPLIT_DIRECTORY)

    # Trained model information
    parser.add_argument('--weights-path', type=str,
                        help='Path to trained tissue segmentation model weights.', required=True)
    parser.add_argument('--segformer-size', type=str, help='Size of SegFormer model',
                        choices=['b0', 'b1', 'b2', 'b3', 'b4', 'b5'],
                        default=DEFAULT_SEGFORMER_SIZE)

    # Inference configuration
    parser.add_argument('--batch-size', type=int, help='Batch size',
                        default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--num-workers', type=int, help='Number of workers',
                        default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--mpp', type=float,
                        help='MPP to scale data to. <=0 = no scaling.', default=DEFAULT_MPP)
    parser.add_argument('--ocm', type=int,
                        help='Output crop margin to use', default=DEFAULT_OCM)

    # If samples of overlaid annotations should also be extracted
    parser.add_argument('--extract-overlays', action=argparse.BooleanOptionalAction,
                        help='Also extract images of annotations overlaid on the images. This will '
                             'output 2x sets of images (both from scaled and cropped versions of '
                             'the tissue images to the equivalent areas in the cell images). '
                             'First, colours overlaid on the image data representing the ARGMAXED '
                             'class prediction (red = Cancer area, blue = background area, '
                             'black = unknown). Second, an RGB \'heatmap\' representing the cancer '
                             'area (not overlaid on the original image). Whiter (\'hotter\') '
                             'areas = higher confidence, darker areas = lower confidence).',
                        default=False)

    # Output directory for generated segmentation masks
    parser.add_argument('--output-directory', type=str,
                        help='Directory to store outputs in.', required=True)

    return parser.parse_args()


def main():
    args = parse_args()

    # Load the metadata.json file
    ocelot_metadata = read_json(os.path.join(args.ocelot_directory, 'metadata.json'))['sample_pairs']

    # Set up output directory (ensuring it is unique to avoid accidental file overwrites)
    out_dir = args.output_directory
    if os.path.isdir(out_dir):
        raise IsADirectoryError(f'Directory {out_dir} exists. Not writing files!')
    create_directory(out_dir)

    # Create overlay directory (if extracting overlays)
    if args.extract_overlays:
        create_directory(os.path.join(out_dir, 'overlays'))

    # ### Prepare for running model inference ###
    # Set up MPP to use
    mpp = args.mpp
    if mpp <= 0:
        mpp = None

    # Set up the device
    device = get_default_device()

    # Set up datasets/dataloader (run over images in both 'train.txt' and 'val.txt')
    dataloaders = []
    for split_filename in ('train.txt', 'val.txt'):
        if os.path.isfile(os.path.join(args.split_directory, split_filename)):
            dataset = TissueDataset(
                args.data_directory, os.path.join(args.split_directory, split_filename),
                transforms=None, samples_per_region=1, tile_size=(512, 512),
                output_crop_margin=args.ocm, scale_to_mpp=mpp, pad_class_name='Background')
            dataloader = DataLoader(
                dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                collate_fn=None, drop_last=False)
            dataloaders.append(dataloader)
    if len(dataloaders) == 0:
        raise RuntimeError(f'No dataloaders created. Ensure \'train.txt\' or \'val.txt\' exists.')

    # Set up model
    model = SegFormer(num_classes=len(TISSUE_CLASSES), size=args.segformer_size, pretrained=False,
                      train_output_crop_margin=0, eval_output_crop_margin=args.ocm,
                      input_with_mask=False)
    model.load_weights(args.weights_path)
    model.to(device)
    model.eval()

    # Set up postprocessors
    postprocessors = [SegSoftmax()]

    # ##################################### Perform Inference  #####################################
    for dataloader_idx, dataloader in enumerate(dataloaders):
        print(f'Running inference over dataloader {dataloader_idx + 1}/{len(dataloaders)}. '
              f'{dataloader.dataset.num_unique_images} images.')
        generate_store_segmentation_masks(
            model, postprocessors, dataloader, device, mpp, ocelot_metadata, out_dir,
            args.extract_overlays, args.data_directory)


def generate_store_segmentation_masks(model, postprocessors, dataloader, device, mpp,
                                      ocelot_metadata, output_directory, extract_overlays,
                                      data_directory):
    # Determine what input keys the model requires
    model_input_keys = model.required_input_keys

    # Store of region ID (image filename) to data for that region
    region_data_store = {}
    loaded_region_paths = []

    progress = tqdm(dataloader)
    for batch_idx, batch in enumerate(progress):
        # ################################ STORES RELATING TO BATCH ################################
        last_batch = batch_idx == len(dataloader) - 1          # Whether this is the last batch

        # ################################ SET UP REGION DATA STORE ################################
        # Determine the number of samples based on the expected input_path key
        num_samples = len(batch['input_path'])
        for sample_idx in range(num_samples):
            # Extract the input path
            input_path = batch['input_path'][sample_idx]
            # Set up the region data store with that input path if it isn't already there
            if input_path not in region_data_store:
                dimensions = batch['dimensions'][sample_idx]
                if isinstance(dimensions, torch.Tensor):
                    dimensions = dimensions.cpu().numpy()
                region_data_store[input_path] = {
                    'id': batch['id'][sample_idx],
                    'dimensions': dimensions,
                    'predictions': [],
                    'prediction_coords': [],
                }
                loaded_region_paths.append(input_path)

            # Append 'prediction_coords' for each sample to the region_data_store
            output_coordinates = batch['output_coordinates'][sample_idx]
            if isinstance(output_coordinates, torch.Tensor):
                output_coordinates = output_coordinates.cpu().numpy()
            output_coordinates = output_coordinates.tolist()
            region_data_store[input_path]['prediction_coords'].append(output_coordinates)

        # Put required data onto GPU
        model_inputs = {key: move_data_to_device(batch[key], device) for key in model_input_keys}

        # Perform forward pass
        with torch.no_grad():
            model_outputs = model.forward(model_inputs)

        # Update store with output predictions
        for sample_idx in range(num_samples):
            # Get the input path
            input_path = batch['input_path'][sample_idx]

            # Update the predictions store for that model with all model outputs
            region_data_store[input_path]['predictions'].append({
                key: model_outputs[key][sample_idx].cpu() for key in model_outputs.keys()
            })

        # ################################ STORE PREDICTIONS (IF READY) ############################
        while len(loaded_region_paths) > 1 or last_batch:
            # Extract the data to store
            path_to_store = loaded_region_paths.pop(0)
            region_data = region_data_store.pop(path_to_store)
            image_id = region_data['id']

            # Collate all individual predictions into a single prediction
            complete_prediction = model.collate_outputs(
                region_data['predictions'], region_data['prediction_coords'],
                region_data['dimensions'])

            # Perform postprocessing on outputs
            complete_prediction = postprocess_outputs(complete_prediction, postprocessors)

            # Extract and store the softmaxed cancer area mask
            extract_store_softmasked_cancer_mask(
                complete_prediction, image_id, mpp, ocelot_metadata, output_directory,
                extract_overlays, path_to_store, data_directory)

            # Stop iteration if this was triggered due to the last batch
            if len(loaded_region_paths) == 0 and last_batch:
                break


def extract_store_softmasked_cancer_mask(complete_prediction, image_id, mask_mpp, ocelot_metadata,
                                         output_directory, extract_overlays, rel_image_path,
                                         data_directory):
    # Extract the softmaxed key from model outputs (C, H, W)
    sm_seg_mask = complete_prediction[SEG_MASK_PROB_KEY].detach().cpu().numpy()

    # Extract the softmaxed heatmap for the 'cancer area' class. Giving (H, W)
    cancer_area_hm = sm_seg_mask[TISSUE_CLASSES.index('Cancer')]

    # Determine how to scale/crop mask to get to the cell level
    image_metadata = ocelot_metadata[image_id]
    (sf_x, sf_y), crop_info = cell_scale_crop_in_tissue_at_cell_mpp(
        meta_pair=image_metadata, tissue_mpp=mask_mpp, cell_mpp=None)

    # Scale the mask
    new_w, new_h = int(round(cancer_area_hm.shape[1] * sf_x)), int(round(cancer_area_hm.shape[0] * sf_y))
    cancer_area_hm = cv2.resize(cancer_area_hm, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Crop the scaled mask
    cancer_area_hm = crop_image(cancer_area_hm, crop_info)

    # Ensure the mask is at the right shape
    assert cancer_area_hm.shape[0] == OUTPUT_MASK_SHAPE and cancer_area_hm.shape[1] == OUTPUT_MASK_SHAPE

    # Write the mask to disk (in output directory. Name = <image_id>_ca_hm.npy)
    tif_profile = DEFAULT_TIF_PROFILE.copy()
    tif_profile['count'] = 1
    tif_profile['width'] = cancer_area_hm.shape[1]
    tif_profile['height'] = cancer_area_hm.shape[0]
    tif_profile['dtype'] = np.float32
    write_tif_rasterio(
        os.path.join(output_directory, f'{image_id}_ca_hm.tif'), cancer_area_hm, tif_profile,
        overwrite=True, pyramid_factors=[2, 4, 8], resampling=Resampling.average)

    # Also overlay the mask on the original image if requested
    if extract_overlays:
        # ### Argmaxed colours ###
        # Load the image (at original MPP)
        tissue_image, _ = load_tif_rasterio(os.path.join(data_directory, rel_image_path))
        # Determine how to scale and crop the original image to cell area
        (sf_x_im, sf_y_im), crop_info_im = cell_scale_crop_in_tissue_at_cell_mpp(
            meta_pair=image_metadata, tissue_mpp=None, cell_mpp=None)
        new_w_im, new_h_im = int(round(tissue_image.shape[1] * sf_x_im)), int(round(tissue_image.shape[0] * sf_y_im))
        # Scale and crop the image (to get equivalent cell area)
        tissue_image = cv2.resize(tissue_image, (new_w_im, new_h_im), interpolation=cv2.INTER_AREA)
        tissue_image = crop_image(tissue_image, crop_info_im)
        # Take argmax of original mask and find equivalent area
        am_seg_mask = torch.argmax(complete_prediction[SEG_MASK_PROB_KEY], dim=0).detach().cpu().numpy().astype(np.uint8)
        am_seg_mask = cv2.resize(am_seg_mask, (new_w, new_h), interpolation=cv2.INTER_AREA)
        am_seg_mask = crop_image(am_seg_mask, crop_info)
        # Generate RGB mask
        tissue_am_overlay = np.zeros_like(tissue_image)
        for cls_idx, cls_colour in enumerate(TISSUE_CLASS_COLOURS):
            tissue_am_overlay[am_seg_mask == cls_idx] = cls_colour
        # Blend with original image
        tissue_am_overlay = overlay_images(tissue_image, tissue_am_overlay, 0.25)
        # Write to disk
        write_image(os.path.join(output_directory, 'overlays', f'{image_id}_ca_am.jpg'),
                    tissue_am_overlay, overwrite=True)

        # ### RGB heatmap ###
        rgb_heatmap = cv2.cvtColor(
            cv2.applyColorMap(
                (cancer_area_hm * 255).astype(np.uint8), cv2.COLORMAP_HOT),
            cv2.COLOR_BGR2RGB)
        write_image(os.path.join(output_directory, 'overlays', f'RGB_heatmap_{image_id}_ca.jpg'),
                    rgb_heatmap, overwrite=True)


if __name__ == '__main__':
    main()
