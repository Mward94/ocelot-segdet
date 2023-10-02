"""Torch dataset used for loading tissue images with segmentation ground truth.
"""
import os
from typing import Optional, Union, Tuple, List

import albumentations as A
import torch
from torchvision.transforms import functional as F

from datasets.ocelot_dataset import OcelotDataset, TILE_COORDS
from util.constants import TISSUE_CLASSES, GT_SEG_MASK_KEY, INPUT_IMAGE_KEY, INPUT_MPP_KEY
from util.helpers import calculate_cropped_size, get_region_dimensions, convert_coordinates_to_mpp
from util.image import load_tif_rasterio


class TissueDataset(OcelotDataset):
    def __init__(self, data_directory: str, split_filepath: str,
                 transforms: Optional[List[A.BasicTransform]] = None, samples_per_region: int = 1,
                 tile_size: Optional[Union[int, Tuple[int, int]]] = None,
                 output_crop_margin: Optional[int] = None, scale_to_mpp: Optional[float] = None,
                 pad_class_name: Optional[str] = None):
        """Constructor for dataset tailored to tissue segmentation.

        Args:
            data_directory: Path to directory containing processed data.
            split_filepath: Path to the .txt file describing cases belonging to this dataset.
            transforms: A list of albumentations transforms that should be applied to the image
                data. ToFloat() and ToTensor() transforms are always applied at the end.
            samples_per_region: The number of times a region should be sampled from in a batch.
            tile_size: The desired size of tiles if tiling regions. If an int is given, the tiles
                will be square. If a tuple, tiles with that height and width will be created. NOTE:
                If an image is smaller than the tile size, padding will be applied to the image.
                Tiling occurs BEFORE transforms, etc. are made.
            output_crop_margin: The output crop margin reduces the output size of the ground truth
                data this dataset generates relative to the input size. If also tiling, the
                generated tiles will be strided by the derived output size.
            scale_to_mpp: If specified, images will be scaled as if observed in a different mpp
                resolution (provided mpp_x and mpp_y in patient_data).
            pad_class_name: The name of the post-mapping class which should be used to label pixels
                in the segmentation mask which constitute padding. This parameter must be specified
                when loading image data outside the image bounds or using transforms which
                introduce zero-padding.
        """
        super().__init__('tissue', data_directory, split_filepath, transforms, None,
                         samples_per_region, tile_size, output_crop_margin, scale_to_mpp)

        # Store the name of the class to use as 'background'
        if pad_class_name is not None and pad_class_name not in TISSUE_CLASSES:
            raise ValueError(f'Expected pad_class_name={pad_class_name} to be in '
                             f'class_list={TISSUE_CLASSES}.')
        self.pad_class_name = pad_class_name

    def __getitem__(self, idx):
        """Gets the next element from the dataset at index <idx>

        Args:
            idx (int): Index of the element to retrieve data for

        Returns:
            (dict): A dictionary of data belonging to the requested sample. This ground truth
                included with this dictionary is comprised of the constant GT_SEG_MASK_KEY, which
                represents the ground truth segmentation mask.
        """
        region_data = self.all_region_data[idx]
        example = self._load_item(region_data)

        # Crop the ground truth mask if output_crop_margin is defined
        margin = self.output_crop_margin
        if margin is not None:
            height, width = calculate_cropped_size(example[GT_SEG_MASK_KEY].shape[-2:], margin)
            example[GT_SEG_MASK_KEY] = F.crop(example[GT_SEG_MASK_KEY],
                                              top=margin, left=margin, height=height, width=width)

        return example

    def _load_item(self, region_data: dict, disable_tiling: bool = False):
        # If tiling, determine the window to load
        load_window = None
        out_size = region_data['dimensions']
        if not disable_tiling and TILE_COORDS in region_data:
            tile_window = region_data[TILE_COORDS]
            load_window = convert_coordinates_to_mpp(       # Tile window at scaled MPP
                tile_window, self.scale_to_mpp, region_data['mpp'], round_int=True)
            out_size = get_region_dimensions(tile_window)

        # Load the input region
        input_region, _ = load_tif_rasterio(
            os.path.join(self.data_directory, region_data['image_path']), window=load_window,
            out_size=out_size)

        # Load the ground truth
        load_fill_value = TISSUE_CLASSES.index(self.pad_class_name)  # Fill value to use for out of bounds pixels when reading the mask
        ground_truth, _ = load_tif_rasterio(
            os.path.join(self.data_directory, region_data['gt_path']), window=load_window,
            out_size=out_size, resampling='nearest', fill_value=load_fill_value)

        # Apply any transforms.
        # Zero padding may appear during the transform. To avoid this, we first increase the class
        # index of each GT class by 1 to ensure no class has index zero, this means we can be sure
        # that all index 0 elements are padding.
        ground_truth += 1

        transformed = self.transform(image=input_region, mask=ground_truth)
        input_region = transformed['image']
        ground_truth = transformed['mask']

        # Map back from +1 indices to indices in class_list, handling 0 elements.
        pad_indices = torch.eq(ground_truth, 0)
        if torch.any(pad_indices):
            if self.pad_class_name is None:
                raise RuntimeError('Padding was introduced while loading and transforming data, '
                                   'but pad_class_name is not specified.')
            ground_truth[pad_indices] = TISSUE_CLASSES.index(self.pad_class_name) + 1
        ground_truth -= 1

        # Return data, along with some other useful metadata
        used_mpp = (self.scale_to_mpp, self.scale_to_mpp) if self.scale_to_mpp is not None else region_data['mpp']

        return {
            'id': region_data['id'],
            'input_path': region_data['image_path'],
            'dimensions': region_data['dimensions'],
            'output_coordinates': self.get_output_coordinates(input_region, region_data, offset_to_tile=True),
            INPUT_MPP_KEY: used_mpp,
            INPUT_IMAGE_KEY: input_region,
            GT_SEG_MASK_KEY: ground_truth,
        }

    def get_complete_input_and_ground_truth_for_input_path(self, input_path):
        """Returns complete input and ground truth data corresponding to a given input path

        Ensures any transforms/scaling is still applied to the input/ground truth
        Any tiling/output size specifications should be ignored

        Args:
            input_path (str): Path to the input

        Returns:
            (dict): A dictionary of input and ground truth data belonging to the requested sample.
                It is unnecessary to include any metadata with this
        """
        region_data = self.get_region_data_for_input_path(input_path)
        return self._load_item(region_data, disable_tiling=True)
