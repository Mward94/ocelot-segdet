"""Torch dataset used for loading cell images with detection ground truth.

This includes also loading the softmaxed cancer area mask.
"""
import os
from typing import Optional, List, Union, Tuple, Dict, Any

import albumentations as A
import numpy as np
import torch
from numpy.typing import NDArray
from torchvision.transforms import functional as F

from datasets.ocelot_dataset import OcelotDataset, TILE_COORDS
from util.constants import INPUT_IMAGE_KEY, GT_POINT_HEATMAP_KEY, GT_POINTS_KEY, GT_INDICES_KEY, \
    INPUT_MASK_PROB_KEY, INPUT_IMAGE_MASK_KEY, CELL_CLASSES, INPUT_MPP_KEY
from util.helpers import calculate_cropped_size, point_intersects_region, \
    convert_coordinates_to_mpp, get_region_dimensions, convert_pixel_mpp
from util.image import load_tif_rasterio, generate_gaussian_point_heatmap


class CellDataset(OcelotDataset):
    def __init__(self, data_directory: str, split_filepath: str,
                 transforms: Optional[List[A.BasicTransform]] = None, samples_per_region: int = 1,
                 tile_size: Optional[Union[int, Tuple[int, int]]] = None,
                 output_crop_margin: Optional[int] = None, scale_to_mpp: Optional[float] = None,
                 gaussian_sigma: float = 1.14, gaussian_sigma_units: str = 'microns',
                 seg_mask_dir: Optional[str] = None):
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
            gaussian_sigma: Standard deviation of Gaussian used to generate ground truth
                annotations.
            gaussian_sigma: When generating the target heatmap, the standard deviation of the
                Gaussians centred on each point.
            gaussian_sigma_units: Either 'microns' or 'pixels'. Dictates whether the standard
                deviation is specified in microns or pixels.
            seg_mask_dir: If specified, looks for segmentation masks in this directory. These are
                loaded and concatenated with the input RGB image. Expects filenames to follow
                convention: <image_id>_ca_hm.tif.
        """
        transform_params = dict(keypoint_params=A.KeypointParams(
            format='xy', label_fields=['labels'], remove_invisible=True))

        super().__init__('cell', data_directory, split_filepath, transforms, transform_params,
                         samples_per_region, tile_size, output_crop_margin, scale_to_mpp)

        # Store how to define the Gaussian
        if gaussian_sigma_units not in {'pixels', 'microns'}:
            raise ValueError(f'Gaussian sigma units must be \'pixels\' or \'microns\'. '
                             f'Was: {gaussian_sigma_units}')
        self.gaussian_sigma = gaussian_sigma
        self.gaussian_sigma_units = gaussian_sigma_units

        # Store the segmentation mask directory to use
        self.seg_mask_dir = seg_mask_dir

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

        # Handle the output size (if specified)
        #   Handled by:
        #       Cropping the heatmap
        #       Removing any points that don't intersect the output region
        #       Shifting all remaining ground truth points to be relative to output size
        if self.output_crop_margin is not None:
            gt_heatmap, gt_points, gt_labels = self._shift_ground_truth_to_output_region(
                example[INPUT_IMAGE_KEY], example[GT_POINT_HEATMAP_KEY],
                example[GT_POINTS_KEY], example[GT_INDICES_KEY])
            example[GT_POINT_HEATMAP_KEY] = gt_heatmap
            example[GT_POINTS_KEY] = gt_points
            example[GT_INDICES_KEY] = gt_labels

        if self.seg_mask_dir is not None:
            # Given these masks are at input, the output crop margin does NOT apply
            # Concatenate the ground truth seg mask onto the RGB image data
            # Mask stored as (H, W). So unsqueeze to enable concat
            example[INPUT_IMAGE_MASK_KEY] = torch.cat([
                example[INPUT_IMAGE_KEY],
                example[INPUT_MASK_PROB_KEY].unsqueeze(0),
            ], dim=0)
        return example

    def _load_item(self, region_data: dict, disable_tiling: bool = False):
        # If tiling, determine the window to load
        load_window = None
        out_size = region_data['dimensions']
        if not disable_tiling and TILE_COORDS in region_data:
            tile_window = region_data[TILE_COORDS]
            load_window = convert_coordinates_to_mpp(  # Tile window at scaled MPP
                tile_window, self.scale_to_mpp, region_data['mpp'], round_int=True)
            out_size = get_region_dimensions(tile_window)

        # Load the input region
        input_region, crop_window = load_tif_rasterio(
            os.path.join(self.data_directory, region_data['image_path']), window=load_window,
            out_size=out_size)

        # Load the ground truth (this also handles class mapping)
        gt_points, gt_labels = self._load_cell_annotations(region_data, window=load_window)

        # If scaling to mpp, scale the ground truth
        if self.scale_to_mpp is not None:
            orig_mpp_x, orig_mpp_y = region_data['mpp']
            gt_points = np.stack([
                convert_pixel_mpp(gt_points[:, 0], orig_mpp_x, self.scale_to_mpp),
                convert_pixel_mpp(gt_points[:, 1], orig_mpp_y, self.scale_to_mpp),
            ], axis=-1)

        # Set up where to bound the point annotations to (based on crop_window)
        #   If no crop_window, is set as the whole image
        if crop_window is not None:
            bound_to_region = crop_window
        else:
            bound_to_region = [0, 0, input_region.shape[1], input_region.shape[0]]

        # Bound the ground truth point annotations (points) to the valid region
        # Any points outside of the image area are removed
        gt_points, gt_labels = self._bound_cell_annotations_to_region(
            gt_points, gt_labels, bound_to_region)

        # Load the seg mask (if required)
        extra_return_kwargs = {}
        extra_tx_kwargs = {}
        if self.seg_mask_dir is not None:
            seg_mask_path = os.path.join(self.seg_mask_dir, f'{region_data["id"]}_ca_hm.tif')

            # Loaded as (H, W) - single-channel heatmap for cancer area
            seg_ground_truth, _ = load_tif_rasterio(
                seg_mask_path, window=load_window, out_size=out_size, resampling='average',
                fill_value=0, dtype=np.float32)

            if seg_ground_truth.ndim != 2:
                raise ValueError(f'Expected single softmaxed mask (2 dims). Found mask with '
                                 f'dimensionality {seg_ground_truth.shape}')

            extra_tx_kwargs['mask'] = seg_ground_truth

        # Apply any transforms
        transformed = self.transform(image=input_region, keypoints=gt_points, labels=gt_labels,
                                     **extra_tx_kwargs)
        input_region = transformed['image']
        gt_points, gt_labels = np.asarray(transformed['keypoints']), np.asarray(transformed['labels'])
        if self.seg_mask_dir is not None:
            seg_ground_truth = transformed['mask']
            extra_return_kwargs[INPUT_MASK_PROB_KEY] = seg_ground_truth

        # Handle no ground truth points (set to a [0, 2] array)
        if len(gt_points) == 0:
            gt_points, gt_labels = np.zeros([0, 2], dtype=np.float32), np.zeros(0, dtype=np.int64)

        # Generate the ground truth Gaussian (creates a Tensor)
        used_mpp = (self.scale_to_mpp, self.scale_to_mpp) if self.scale_to_mpp is not None else region_data['mpp']
        gt_point_heatmap = self.generate_gaussian(input_region, gt_points, gt_labels, mpp=used_mpp)

        # Put ground truth points/labels into torch tensor with relevant datatypes
        gt_points = torch.as_tensor(gt_points, dtype=torch.float32)
        gt_labels = torch.as_tensor(gt_labels, dtype=torch.long)

        return {
            'id': region_data['id'],
            'input_path': region_data['image_path'],
            'dimensions': region_data['dimensions'],
            'output_coordinates': self.get_output_coordinates(input_region, region_data, offset_to_tile=True),
            INPUT_MPP_KEY: used_mpp,
            INPUT_IMAGE_KEY: input_region,
            GT_POINT_HEATMAP_KEY: gt_point_heatmap,
            GT_POINTS_KEY: gt_points,
            GT_INDICES_KEY: gt_labels,
            **extra_return_kwargs,
        }

    def _load_cell_annotations(
            self,
            region_data: Dict[str, Any],
            window: Union[Tuple[int, int, int, int], List[int]] = None,
    ) -> Tuple[NDArray, NDArray]:
        """Loads and prepares point cell annotations in a consistent form

        This handles performing any required class mapping

        Args:
            region_data: Dictionary containing data for a region.
            window: A list of length 4 with the x1, y1, x2, y2 coordinates that annotations should
                be shifted within.

        Returns:
            The loaded ground truth points with shape: (N, 2)
            The corresponding ground truth class indices with shape: (N,).
        """
        # Stores for ground truth coordinates and classes
        gt_coords, gt_labels = [], []

        # Load annotations from all cell classes
        for cell_class, annotation_path in region_data['gt_path'].items():
            # Determine the index of the class (index into the class list). This does the mapping
            class_idx = CELL_CLASSES.index(cell_class)

            # Load the point annotations
            coords = np.load(os.path.join(self.data_directory, annotation_path))

            # Shift annotations based on window to be loaded
            if window is not None:
                coords[:, 0] -= window[0]
                coords[:, 1] -= window[1]

            # Update ground truth stores
            gt_coords.append(coords)
            gt_labels.extend([class_idx] * len(coords))

        # Collate annotations
        if len(gt_coords) > 0:      # Handle no annotations found
            gt_coords = np.concatenate(gt_coords, axis=0)
        else:
            gt_coords = np.zeros([0, 2], dtype=np.float32)
        gt_labels = np.asarray(gt_labels)

        # Return annotations
        return gt_coords, gt_labels

    def generate_gaussian(
        self,
        input_region: Union[NDArray[np.uint8], torch.Tensor],
        gt_points: NDArray,
        gt_labels: NDArray,
        mpp: Optional[Tuple[float, float]] = None,
    ) -> torch.Tensor:
        """Generates a (Nc x H x W) heatmap with Gaussians centred on each point.

        The generated heatmap contains floats in the range [0, 1].

        mpp: Specify when specifying sigma in microns
        """
        # Extract the size of the input region
        if isinstance(input_region, torch.Tensor):
            h, w = input_region.shape[1:]
        elif isinstance(input_region, np.ndarray):
            h, w = input_region.shape[:2]
        else:
            raise ValueError(f'Type of input region must be a numpy array or torch tensor. '
                             f'Is: {type(input_region)}')

        # Create the target heatmap (Nc x H x W)
        heatmap = np.zeros((len(CELL_CLASSES), h, w), dtype=np.float32)

        # Determine size of Gaussian
        if self.gaussian_sigma_units == 'pixels':
            sigma = self.gaussian_sigma
        elif self.gaussian_sigma_units == 'microns':
            if mpp is None:
                raise ValueError('Requires mpp when sigma units is microns.')
            if mpp[0] != mpp[1]:
                raise NotImplementedError(f'Must have equal MPP in x/y axis. Mpp is: {mpp}.')
            sigma = self.gaussian_sigma / mpp[0]
        else:
            raise RuntimeError(f'{self.gaussian_sigma_units}')

        # Per-class, extract the points for that class and set up Gaussians
        for cls_idx in range(len(CELL_CLASSES)):
            label_idxs = np.where(gt_labels == cls_idx)[0]
            cls_points = gt_points[label_idxs]
            if len(cls_points) > 0:
                generate_gaussian_point_heatmap(
                    points=cls_points,
                    sigma=sigma,
                    out_heatmap=heatmap[cls_idx],
                )

        # Convert heatmap into a torch tensor
        heatmap = torch.as_tensor(heatmap)

        return heatmap

    @staticmethod
    def _bound_cell_annotations_to_region(gt_points, gt_labels, region_coords):
        """Ensures all cell point annotations are bound to the given region coordinates

        Args:
            gt_points (ndarray): The ground truth points
            gt_labels (ndarray): The ground truth class indices
            region_coords ([int]): The x1, y1, x2, y2 coordinates of the region to bound
                annotations within

        Returns:
            (tuple):
                * (np.ndarray): The ground truth points bounded to the region
                * (np.ndarray): The corresponding ground truth class indices
        """
        # Determine which indices are within the region
        within_region_indices = np.flatnonzero(
            ((gt_points[:, 0] >= region_coords[0]) & (gt_points[:, 1] >= region_coords[1]) &
             (gt_points[:, 0] < region_coords[2]) & (gt_points[:, 1] < region_coords[3])))

        # Return ground truth with instances removed
        return gt_points[within_region_indices, :], gt_labels[within_region_indices]

    def _shift_ground_truth_to_output_region(self, input_region, gt_heatmap, gt_points, gt_labels):
        """Given loaded ground truth, shifts them relative to output region

        If any points are found that no longer intersect the output region, those points are removed
        These points are not removed from the heatmap

        Args:
            input_region (ndarray/torch.Tensor): The loaded input region, used to determine the
                coordinates of the output region
            gt_heatmap (torch.Tensor): The ground truth point target heatmap
            gt_points (torch.Tensor): The ground truth points
            gt_labels (torch.Tensor): The ground truth class indices

        Returns:
            (tuple):
                * (torch.Tensor): The cropped ground truth heatmap
                * (torch.Tensor): The ground truth points
                * (torch.Tensor): The corresponding ground truth class indices
        """
        # Get the size of the input region
        if isinstance(input_region, torch.Tensor):
            input_height, input_width = input_region.shape[1:]
        else:
            input_height, input_width = input_region.shape[:2]

        # Determine top-left coords of output region
        output_height, output_width = calculate_cropped_size(
            (input_height, input_width), self.output_crop_margin)

        # Crop the heatmap
        gt_heatmap = F.crop(gt_heatmap, top=self.output_crop_margin, left=self.output_crop_margin,
                            height=output_height, width=output_width)

        # Set coords as x1, y1, x1 + width, y1 + height
        output_coords = [self.output_crop_margin, self.output_crop_margin,
                         self.output_crop_margin + output_width, self.output_crop_margin + output_height]

        # Filter out any points that don't intersect with output_coords box
        points_intersect = [point_intersects_region(output_coords, point.numpy()) for point in gt_points]
        gt_points = gt_points[points_intersect]
        gt_labels = gt_labels[points_intersect]

        # Shift points to treat output_coords as (0, 0) (If there are any points)
        if len(gt_points) > 0:
            gt_points[:, 0] -= output_coords[0]
            gt_points[:, 1] -= output_coords[1]

        return gt_heatmap, gt_points, gt_labels

    def get_complete_input_and_ground_truth_for_input_path(self, input_path):
        """Returns complete input and ground truth data corresponding to a given input path

        This should ensure that any transforms/scaling is still applied to the input/ground truth
        Any tiling/output size specifications should be ignored

        Args:
            input_path (str): Path to the input

        Returns:
            (dict): A dictionary of input and ground truth data belonging to the requested sample.
                It is unnecessary to include any metadata with this
        """
        region_data = self.get_region_data_for_input_path(input_path)
        example = self._load_item(region_data, disable_tiling=True)

        if self.seg_mask_dir is not None:
            # Given these masks are at input, the output crop margin does NOT apply
            # Concatenate the ground truth seg mask onto the RGB image data
            # Mask stored as (H, W). So unsqueeze to enable concat
            example[INPUT_IMAGE_MASK_KEY] = torch.cat([
                example[INPUT_IMAGE_KEY],
                example[INPUT_MASK_PROB_KEY].unsqueeze(0),
            ], dim=0)
        return example
