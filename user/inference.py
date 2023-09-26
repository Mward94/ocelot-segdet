import os
from typing import Tuple, List, Dict, Any, Union, Iterable, Optional

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from torch.utils.data import DataLoader

from datasets.image_dataset import ImageDataset
from networks.postprocessors.gaussian_modulation import GaussianModulation
from networks.postprocessors.point_heatmap_blob_detector import PointHeatmapBlobDetector
from networks.postprocessors.point_nms import PointNMS
from networks.postprocessors.postprocessor import Postprocessor
from networks.postprocessors.seg_mask_to_point_heatmap import SegMaskToPointHeatmap
from networks.postprocessors.seg_softmax import SegSoftmax
from networks.segformer import SegFormer
from util.constants import (
    INPUT_IMAGE_KEY, INPUT_IMAGE_MASK_KEY, SEG_MASK_PROB, DET_POINTS_KEY, DET_INDICES_KEY,
    DET_SCORES_KEY)
from util.helpers import calculate_cropped_size, convert_dimensions_to_mpp, scale_coords_to_mpp
from util.image import precompute_macenko_params, crop_image
from util.ocelot_parsing import get_region_mpp, cell_scale_crop_in_tissue_at_cell_mpp
from util.tiling import generate_tiles
from util.torch import get_default_device, move_data_to_device

# ### Constants ###
# General constants
NUM_WORKERS = 0
TILE_SIZE = (512, 512)
TISSUE_MASK_CANCER_AREA_IDX = 1     # Cancer area index for tissue segmentation mask

# Model weights paths
TISSUE_WEIGHTS_PATH = 'tissue_model_weights.pth'
CELL_WEIGHTS_PATH = 'cell_model_weights.pth'

# Tissue model configuration
TISSUE_MODEL_BATCH_SIZE = 8
TISSUE_MODEL_MACENKO = True
TISSUE_MODEL_MPP = 0.5
TISSUE_MODEL_OCM = 64
TISSUE_MODEL_SIZE = 'b0'

# Cell model configuration
CELL_MODEL_BATCH_SIZE = 8
CELL_MODEL_MACENKO = False
CELL_MODEL_MPP = 0.2
CELL_MODEL_OCM = 128
CELL_MODEL_SIZE = 'b2'

# Postprocessors to apply to model outputs
TISSUE_POSTPROCESSORS = [SegSoftmax()]
CELL_POSTPROCESSORS = [
    SegMaskToPointHeatmap(),
    GaussianModulation(sigma=5.714),    # Based on 8um diameter ((8/7)/.2 ~= 5.714). 7stds, .2MPP
    PointHeatmapBlobDetector(ignore_index=0),   # Ignore 'Background' class
    PointNMS(euc_dist_threshold=15),
]


class Model():
    """
    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """
    def __init__(self, metadata):
        self.metadata = metadata

        # Get the device to use
        self.device = get_default_device()

        # Get reference to the location of this file (to know where weights are)
        self._curr_path = os.path.split(__file__)[0]

        # Instantiate the tissue segmentation model, load model weights, put on device and set eval
        self.tissue_seg_model = SegFormer(
            num_classes=3, size=TISSUE_MODEL_SIZE, eval_output_crop_margin=TISSUE_MODEL_OCM)
        self.tissue_seg_model.load_weights(
            os.path.join(self._curr_path, 'weights', TISSUE_WEIGHTS_PATH))
        self.tissue_seg_model.to(self.device)
        self.tissue_seg_model.eval()

        # Instantiate the cell segmentation model, load model weights, put on device and set eval
        self.cell_det_model = SegFormer(
            num_classes=3, size=CELL_MODEL_SIZE, eval_output_crop_margin=CELL_MODEL_OCM,
            input_with_mask=True, mask_channels=1)
        self.cell_det_model.load_weights(
            os.path.join(self._curr_path, 'weights', CELL_WEIGHTS_PATH))
        self.cell_det_model.to(self.device)
        self.cell_det_model.eval()

    def __call__(self, cell_patch, tissue_patch, pair_id):
        """This function detects the cells in the cell patch. Additionally
        the broader tissue context is provided.

        NOTE: this implementation offers a dummy inference example. This must be
        updated by the participant.

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8]
            Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        # Getting the metadata corresponding to the patch pair ID
        meta_pair = self.metadata[pair_id]

        # Extract the cell, tissue, and WSI MPP (x, y)
        cell_mpp = get_region_mpp(meta_pair, 'cell')
        tissue_mpp = get_region_mpp(meta_pair, 'tissue')

        # Compute Macenko parameters on the tissue patch (apply to cell patch later)
        if TISSUE_MODEL_MACENKO or CELL_MODEL_MACENKO:
            try:
                macenko_params = precompute_macenko_params(tissue_patch)
            except Exception:
                # If any errors in Macenko, don't normalise image (enabling algorithm to still run)
                macenko_params = None
        else:
            macenko_params = None
        tissue_macenko_params = macenko_params if TISSUE_MODEL_MACENKO else None
        cell_macenko_params = macenko_params if CELL_MODEL_MACENKO else None

        # Perform tissue segmentation (done at TISSUE_MPP)
        tissue_mask = tissue_segmentation(
            tissue_patch, tissue_mpp, self.tissue_seg_model, postprocessors=TISSUE_POSTPROCESSORS,
            req_mpp=TISSUE_MODEL_MPP, tile_size=TILE_SIZE, crop_margin=TISSUE_MODEL_OCM,
            macenko_params=tissue_macenko_params, batch_size=TISSUE_MODEL_BATCH_SIZE,
            num_workers=NUM_WORKERS, device=self.device)

        # Extract the cancer area channel from the tissue mask
        tissue_mask = tissue_mask[[TISSUE_MASK_CANCER_AREA_IDX], ...]

        # Determine where the cell patch is relative to the tissue mask
        (sf_x, sf_y), tissue_cell_crop_coords = cell_scale_crop_in_tissue_at_cell_mpp(
            meta_pair, tissue_mpp=TISSUE_MODEL_MPP, cell_mpp=CELL_MODEL_MPP)

        # Resize the tissue mask to the cell MPP, then crop cell area
        new_width = int(round(tissue_mask.shape[2] * sf_x))
        new_height = int(round(tissue_mask.shape[1] * sf_y))

        # Reshape to (H, W, C) to resize and crop, then back to (C, H, W)
        tissue_mask_cell = np.transpose(tissue_mask, (1, 2, 0))
        tissue_mask_cell = cv2.resize(
            tissue_mask_cell, (new_width, new_height), interpolation=cv2.INTER_AREA)
        # cv2 resize will drop channel dimension. Here we add it back in to get (H, W, C)
        tissue_mask_cell = np.expand_dims(tissue_mask_cell, axis=2)
        tissue_mask_cell = crop_image(tissue_mask_cell, tissue_cell_crop_coords)
        tissue_mask_cell = np.transpose(tissue_mask_cell, (2, 0, 1))

        # Note: Given way model trained, will return 1 = BG cell, 2 = TC
        cell_coords, cell_indices, cell_scores = cell_detection(
            cell_patch, cell_mpp, self.cell_det_model, postprocessors=CELL_POSTPROCESSORS,
            req_mpp=CELL_MODEL_MPP, tile_size=TILE_SIZE, crop_margin=CELL_MODEL_OCM,
            macenko_params=cell_macenko_params, batch_size=CELL_MODEL_BATCH_SIZE,
            num_workers=NUM_WORKERS, device=self.device, input_seg_mask=tissue_mask_cell)

        # Scale cell coordinates back to the cell image MPP
        if len(cell_coords) > 0:
            scaled_cell_coords = scale_coords_to_mpp(
                cell_coords, from_mpp=CELL_MODEL_MPP, to_mpp=cell_mpp)
        else:
            # No cells detected
            scaled_cell_coords = cell_coords.copy()

        final_predictions = []
        for coord, class_id, score in zip(scaled_cell_coords, cell_indices, cell_scores):
            final_predictions.append(
                (int(round(coord[0])), int(round(coord[1])), int(class_id), float(score)))
        return final_predictions


def tissue_segmentation(
        image: NDArray[np.uint8], image_mpp: Tuple[float, float], model: nn.Module,
        postprocessors: List[Postprocessor], req_mpp: float, tile_size: Tuple[int, int],
        crop_margin: int, macenko_params: Dict[str, Any], batch_size: int, num_workers: int,
        device: torch.device,
) -> Union[NDArray[np.uint8], NDArray[np.float32]]:
    """Performs tissue segmentation on an image.

    Returns the softmaxed mask (per-pixel probabilities) representing the Background, Cancer Area,
    and Other Tissue (based on how model was trained).

    All postprocessing occurs after the output is reconstructed.

    Args:
        image: Image to perform segmentation on.
        image_mpp: MPP of the image (tuple: X, Y).
        model: The model to use to perform inference.
        postprocessors: The set of postprocessors to apply on the predictions.
        req_mpp: The MPP the data should be scaled to when performing inference.
        tile_size: Tile sizes to create at req_mpp.
        crop_margin: The crop margin when running inference.
        macenko_params: Macenko normalisation parameters. Used to normalise image data.
        batch_size: The batch size (when tiling, multiple tiles will be created).
        num_workers: Number of workers for the Dataloader. Can set to 0 for small images.
        device: Device to perform inference on.
    """
    # Compute the tile output size
    req_output_size = calculate_cropped_size(tile_size, crop_margin)

    # Determine the size of the image at 'req_mpp' from 'image_mpp' (X, Y)
    required_dimensions = convert_dimensions_to_mpp(
        image.shape[:2][::-1], image_mpp, req_mpp, round_int=True)

    # Extract tile coordinates to sample at req_mpp (Generate at 'tile_size')
    # These all have size 'tile_size'. Note: predictions generated at 'req_output_size'
    tiles = generate_tiles([0, 0, *required_dimensions], tile_size, req_output_size)

    # Create a Dataset to load samples from
    dataset = ImageDataset(
        image, image_mpp, req_mpp, tile_size, tiles, crop_margin,
        macenko_params=macenko_params)

    # Make dataloader (enabling parallelism and batching)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=None, drop_last=False)

    # Store for inference output for this slide
    data_store = {'dimensions': dataset.required_dimensions, 'predictions': [],
                  'prediction_coords': []}

    # Perform inference on image
    for batch in dataloader:
        # Perform inference on the batch
        batch_preds, batch_output_coords = inference_batch(batch, model, device)

        # Update store with predictions
        data_store['predictions'].extend(batch_preds)
        data_store['prediction_coords'].extend(batch_output_coords)

    # Collate all individual predictions into a single prediction
    complete_prediction = model.collate_outputs(
        data_store['predictions'], data_store['prediction_coords'],
        data_store['dimensions'].tolist())

    # Perform any postprocessing on collated version of outputs
    if len(postprocessors) > 0:
        complete_prediction = postprocess_outputs(complete_prediction, postprocessors)

    # Extract and return the integer encoded mask or softmaxed mask
    tissue_mask = complete_prediction[SEG_MASK_PROB]
    return tissue_mask.numpy()


def cell_detection(
        image: NDArray[np.uint8], image_mpp: Tuple[float, float], model: nn.Module,
        postprocessors: List[Postprocessor], req_mpp: float, tile_size: Tuple[int, int],
        crop_margin: int, macenko_params: Dict[str, Any], batch_size: int, num_workers: int,
        device: torch.device, input_seg_mask: Optional[NDArray[np.uint8]] = None,
) -> Union[Tuple[NDArray[np.float32], NDArray[np.int32], NDArray[np.float32]], NDArray[np.float32]]:
    """Performs cell detection (via segmentation) on an image.

    Returns either the detected cell coordinates, classes, and scores; or returns the per-class
    heatmaps (depending on the return_type parameter)

    All postprocessing occurs after the output is reconstructed.

    Args:
        image: Image to perform segmentation on.
        image_mpp: MPP of the image (tuple: X, Y).
        model: The model to use to perform inference.
        postprocessors: The set of postprocessors to apply on the predictions.
        req_mpp: The MPP the data should be scaled to when performing inference.
        tile_size: Tile sizes to create at req_mpp.
        crop_margin: The crop margin when running inference.
        macenko_params: Macenko normalisation parameters. Used to normalise image data.
        batch_size: The batch size (when tiling, multiple tiles will be created).
        num_workers: Number of workers for the Dataloader. Can set to 0 for small images.
        device: Device to perform inference on.
        input_seg_mask: An optional segmentation mask to concatenate with the input.
    """
    # Compute the tile output size
    req_output_size = calculate_cropped_size(tile_size, crop_margin)

    # Determine the size of the image at 'req_mpp' from 'image_mpp' (X, Y)
    required_dimensions = convert_dimensions_to_mpp(
        image.shape[:2][::-1], image_mpp, req_mpp, round_int=True)

    # Extract tile coordinates to sample at req_mpp (Generate at 'tile_size')
    # These all have size 'tile_size'. Note: predictions generated at 'req_output_size'
    tiles = generate_tiles([0, 0, *required_dimensions], tile_size, req_output_size)

    # Create a Dataset to load samples from
    dataset = ImageDataset(
        image, image_mpp, req_mpp, tile_size, tiles, crop_margin, macenko_params=macenko_params,
        input_seg_mask=input_seg_mask)

    # Make dataloader (enabling parallelism and batching)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                            collate_fn=None, drop_last=False)

    # Store for inference output for this slide
    data_store = {'dimensions': dataset.required_dimensions, 'predictions': [],
                  'prediction_coords': []}

    # Perform inference on image
    for batch in dataloader:
        # Perform inference on the batch
        batch_preds, batch_output_coords = inference_batch(batch, model, device)

        # Update store with predictions
        data_store['predictions'].extend(batch_preds)
        data_store['prediction_coords'].extend(batch_output_coords)

    # Collate all individual predictions into a single prediction
    complete_prediction = model.collate_outputs(
        data_store['predictions'], data_store['prediction_coords'],
        data_store['dimensions'].tolist())

    # Perform any postprocessing on collated version of outputs
    if len(postprocessors) > 0:
        complete_prediction = postprocess_outputs(complete_prediction, postprocessors)

    # Based on how model was trained, 1 = background cell. 2 = tumour cell. This is consistent with
    # requirements
    # Extract and return the coordinates, classes, and scores
    return complete_prediction[DET_POINTS_KEY].numpy(), \
        complete_prediction[DET_INDICES_KEY].numpy(), complete_prediction[DET_SCORES_KEY].numpy()


def postprocess_outputs(
        outputs: Dict[str, Any], postprocessors: Iterable[Postprocessor],
) -> Dict[str, Any]:
    """Postprocess model outputs.

    The model outputs will be passed through a chain of postprocessors and the final result
    returned. Additionally, the model output keys will be validated to ensure that each
    postprocessor has all the data it needs.

    Args:
        outputs: The original model outputs.
        postprocessors: A list of postprocessors to apply to the outputs (in order).

    Returns:
        The postprocessed model outputs.
    """
    for postprocessor in postprocessors:
        actual_keys = set(outputs.keys())
        required_keys = set(postprocessor.model_output_keys)
        key_diff = required_keys - actual_keys
        if len(key_diff) > 0:
            key_diff_str = ', '.join(sorted([k for k in key_diff]))
            raise ValueError(f'The {postprocessor.__class__.__name__} postprocessor requires the '
                             f'following missing model outputs: {key_diff_str}')
        outputs = postprocessor.postprocess(outputs)

    return outputs


def inference_batch(batch, model, device):
    """Performs inference on a batch, returning the predictions from the model.
    """
    # Determine the number of images in the batch
    num_images = len(batch[INPUT_IMAGE_KEY])

    # Get the model inputs (and onto correct device)
    model_inputs = {key: move_data_to_device(batch[key], device)
                    for key in (INPUT_IMAGE_KEY, INPUT_IMAGE_MASK_KEY) if key in batch}

    # Perform forward pass
    with torch.no_grad():
        model_outputs = model.forward(model_inputs)

    # Extract predictions and prediction coordinates for the batch
    batch_preds = [{key: model_outputs[key][sample_idx].cpu() for key in model_outputs.keys()}
                   for sample_idx in range(num_images)]
    batch_output_coords = [batch['required_output_coord'][sample_idx].tolist()
                           for sample_idx in range(num_images)]

    return batch_preds, batch_output_coords
