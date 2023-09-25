"""Torch dataset used for loading data with no ground truth annotations.

Supports loading and concatenation of a softmaxed segmentation mask with the input image.
"""
from typing import Tuple, Union, List, Optional, Dict, Any

import cv2
import numpy as np
import torch
from albumentations import Compose, ToFloat
from albumentations.pytorch import ToTensorV2
from numpy.typing import NDArray
from torch.utils.data import Dataset

from util import macenko
from util.constants import INPUT_MASK_PROB_KEY, INPUT_IMAGE_MASK_KEY, INPUT_IMAGE_KEY
from util.helpers import (
    convert_dimensions_to_mpp, crop_image, get_tissue_mask, calculate_cropped_size)


class ImageDataset(Dataset):
    def __init__(
            self, image: NDArray[np.uint8], image_mpp: Tuple[float, float], req_mpp: float,
            req_tile_size: Tuple[int, int],
            req_tiles: Union[List[NDArray[np.int32]], NDArray[np.int32]], req_crop_margin: int,
            macenko_params: Optional[Dict[str, Any]] = None,
            input_seg_mask: Optional[NDArray[np.float32]] = None,
    ):
        """Dataset for loading regions from an image.

        Args:
            image: The image to sample from.
            image_mpp: The MPP of the image.
            req_mpp: The required MPP the data should be loaded at.
            req_tile_size: The size data should be tiled in at req_mpp.
            req_tiles: The collection of tiles that are to be loaded relative to required_mpp.
            req_crop_margin: The crop margin that should be used at req_mpp.
            macenko_params: If any Macenko normalisation is required, these will be passed in to
                the Macenko normalisation. This allows pre-computing normalisation parameters and
                applying them to individual crops.
            input_seg_mask: An additional input mask that will be combined with the input image.
                Will be concatenated along channel dimension.
        """
        # Stores relating to the image and how to sample from it.
        self.image = image
        self.image_mpp = image_mpp
        self.required_mpp = req_mpp
        self.input_seg_mask = input_seg_mask

        # Stores relating to any image normalisation required.
        self.macenko_params = macenko_params

        # Stores relating to tiling.
        self.required_tiles = req_tiles
        self.required_tile_size = req_tile_size
        self.required_crop_margin = req_crop_margin

        # Determine dimensionality of the base level of slide and required level (based on MPP)
        self.base_dimensions = self.image.shape[:2][::-1]  # (W, H)
        self.required_dimensions = convert_dimensions_to_mpp(
            self.base_dimensions, self.image_mpp, self.required_mpp, round_int=True)

        # Given image data is small, resize ahead of time to required_dimensions
        self.resized_image = cv2.resize(image, self.required_dimensions, interpolation=cv2.INTER_AREA)
        if self.input_seg_mask is not None:
            # Incoming as: (C, H, W). Change to: (H, W, C)
            self.input_seg_mask = np.transpose(self.input_seg_mask, (1, 2, 0))

            # Only resize if different MPP
            if self.input_seg_mask.shape[1] != self.required_dimensions[0] or self.input_seg_mask.shape[0] != self.required_dimensions[1]:
                self.resized_input_seg_mask = cv2.resize(self.input_seg_mask, self.required_dimensions, interpolation=cv2.INTER_AREA)
                # cv2 resize will drop channel dimension if there is only 1 axis in the 'channels'
                # dimension. Here we add it back in to get (H, W, C)
                if self.input_seg_mask.ndim == 3 and self.input_seg_mask.shape[2] == 1 and self.resized_input_seg_mask.ndim == 2:
                    self.resized_input_seg_mask = np.expand_dims(self.resized_input_seg_mask, axis=2)
            else:
                self.resized_input_seg_mask = self.input_seg_mask
        else:
            self.resized_input_seg_mask = None

        # Create a ToTensor transform for the images
        self.to_tensor_transform = Compose([ToFloat(), ToTensorV2()])

    def __len__(self):
        return len(self.required_tiles)

    def __getitem__(self, idx):
        """Samples a new tile from the region.
        """
        # Get reference for tile to load
        req_tile = self.required_tiles[idx]

        # Crop from the resized image (pad with black if out of image bounds)
        tile_crop = crop_image(self.resized_image, req_tile, allow_pad=True, pad_fill_value=0)

        # Apply Macenko normalisation
        if self.macenko_params is not None:
            tile_tissue_mask = get_tissue_mask(tile_crop)
            tile_crop = macenko.normalise_he_image(tile_crop, mask=tile_tissue_mask, **self.macenko_params)

        extra_return_kwargs = {}
        extra_tx_kwargs = {}
        if self.resized_input_seg_mask is not None:
            # Extract (H, W, C) masks into [(H, W), ... xC]
            # We pad with 0's (representing 0 probability)
            seg_crop = crop_image(self.resized_input_seg_mask, req_tile, allow_pad=True, pad_fill_value=0)
            extra_tx_kwargs['masks'] = [seg_crop[:, :, i] for i in range(seg_crop.shape[2])]

        # Apply ToTensor transform
        transformed = self.to_tensor_transform(image=tile_crop, **extra_tx_kwargs)
        tile_crop = transformed['image']
        if self.resized_input_seg_mask is not None:
            data_to_cat = [tile_crop]
            if self.resized_input_seg_mask is not None:
                seg_crop = transformed['masks']
                seg_crop = torch.stack(seg_crop, dim=0)
                extra_return_kwargs[INPUT_MASK_PROB_KEY] = seg_crop
                data_to_cat.append(seg_crop)
            extra_return_kwargs[INPUT_IMAGE_MASK_KEY] = torch.cat(data_to_cat, dim=0)

        # Return data
        return {
            INPUT_IMAGE_KEY: tile_crop,
            **extra_return_kwargs,

            # Information on tiling
            'required_tile_coord': req_tile,
            'required_output_coord': self.get_output_coordinates(
                tile_crop, self.required_crop_margin, req_tile, offset_to_tile=True),
        }

    @staticmethod
    def get_output_coordinates(input_region, output_crop_margin, tile_coords=None,
                               offset_to_tile=True):
        """Determines the output coordinates used for that region

        NOTE: By default, the coordinates will be relative to the input region (i.e. (0, 0) is the
            top-left of the input region)

        Args:
            input_region (ndarray/Tensor): The input region following all transforms. If tensor,
                expected ordering is CHW, if ndarray, expected ordering is HWC
            output_crop_margin:
            tile_coords:
            offset_to_tile (bool): If True, the output coordinates are offset by the tile
                coordinates (if found), instead of offset to (0, 0) by default. If no tiling, then
                this does nothing

        Returns:
            (ndarray/None): The expected output coordinates of the input (if applicable), otherwise
                None. If offset_to_tile is True, the coordinates will be relative to the tile
                position in the whole input area, otherwise they will be relative to (0, 0) (i.e.
                the input region)
        """
        # First set up the output coordinates as the shape of the input region
        if isinstance(input_region, torch.Tensor):
            input_height, input_width = input_region.shape[1:]
        else:
            input_height, input_width = input_region.shape[:2]

        # Start by setting the output coords as the whole input area
        output_coords = [0, 0, input_width, input_height]

        # If output_crop_margin specified, output coords should be centre cropped to output_size
        if output_crop_margin is not None:
            output_height, output_width = calculate_cropped_size((input_height, input_width), output_crop_margin)

            # Determine top/left coordinate of output
            output_x1 = int(round((input_width - output_width) / 2))
            output_y1 = int(round((input_height - output_height) / 2))

            # Set coords as x1, y1, x1 + width, y1 + height
            output_coords = [output_x1, output_y1,
                             output_x1 + output_width, output_y1 + output_height]

        # Convert output coordinates to a numpy array
        output_coords = np.asarray(output_coords)

        # Shift the output coords based on TILE_COORDS if required
        if offset_to_tile and tile_coords is not None:
            # Shift the output coords based on tile coords
            output_coords[[0, 2]] += tile_coords[0]
            output_coords[[1, 3]] += tile_coords[1]

        return output_coords
