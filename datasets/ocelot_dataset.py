"""Superclass for tissue and cell datasets.

Implements common methods between the two classes.
"""
import copy
import os
import pickle
from typing import Optional, List, Union, Tuple, Dict, Any

import albumentations as A
import numpy as np
import torch
from albumentations import ToFloat
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from util.helpers import convert_pixel_mpp, calculate_cropped_size
from util.tiling import generate_tiles

# Region data key used to show we are tiling.
TILE_COORDS = '_TILE_COORDINATES'


class OcelotDataset(Dataset):
    def __init__(self, dataset_type: str, data_directory: str, split_filepath: str,
                 transforms: Optional[List[A.BasicTransform]] = None,
                 transform_params: Optional[Dict[str, Any]] = None, samples_per_region: int = 1,
                 tile_size: Optional[Union[int, Tuple[int, int]]] = None,
                 output_crop_margin: Optional[int] = None, scale_to_mpp: Optional[float] = None):
        """Constructor for superclass dataset.

        Args:
            dataset_type: The type of dataset (either 'tissue' or 'cell')
            data_directory: Path to directory containing processed data.
            split_filepath: Path to the .txt file describing cases belonging to this dataset.
            transforms: A list of albumentations transforms that should be applied to the image
                data. ToFloat() and ToTensor() transforms are always applied at the end.
            transform_params: The parameters to supply albumentations with to configure the
                transform pipeline.
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
        """
        if dataset_type not in ('cell', 'tissue'):
            raise RuntimeError(f'Valid dataset types are: \'cell\' or \'tissue\'. '
                               f'Specified: {dataset_type}')
        self.dataset_type = dataset_type

        super().__init__()

        # Store reference to where processed data is stored and the split filepath to use
        self.data_directory = data_directory
        self.split_filepath = split_filepath

        # Store the number of times to sample a region
        self.samples_per_region = samples_per_region

        # Store the tile_size and output_crop_margin and set them correctly
        self.tile_size = tile_size
        if self.tile_size is not None:
            if isinstance(self.tile_size, int):
                self.tile_size = (self.tile_size, self.tile_size)
        self.output_crop_margin = output_crop_margin

        # Store the mpp resolution data should be rescaled to
        self.scale_to_mpp = scale_to_mpp

        # Compose transforms
        if transform_params is None:
            transform_params = {}
        standard_transforms = [ToFloat(), ToTensorV2()]
        if transforms is None:
            self.transform = A.Compose(standard_transforms, **transform_params)
        else:
            self.transform = A.Compose([*transforms, *standard_transforms], **transform_params)

        # Set up store of annotation metadata
        self.all_region_data, self.num_unique_images = self._load_from_split()

    def _load_from_split(self):
        """Loads all data given the filepath to a split to load

        Returns:
            (list): A list of data representing regions that can be sampled from in this dataset
        """
        # Get the list of patient pickle files
        with open(self.split_filepath, 'r') as f:
            metadata_pkl_files = f.read().splitlines()
        metadata_pkl_files = [os.path.join(self.data_directory, 'metadata', f'{line}.pkl')
                              for line in metadata_pkl_files]

        # Load all pickled data
        all_metadata = []
        for file in metadata_pkl_files:
            with open(file, 'rb') as pkl_file:
                all_metadata.append(pickle.load(pkl_file))
        num_unique_images = len(all_metadata)

        # Create a datastore by region
        all_region_data = []

        # Populate the datastore
        for metadata in all_metadata:
            region_data = metadata[self.dataset_type]
            region_data['id'] = metadata['id']
            region_data['original_dimensions'] = np.asarray(region_data['dimensions'])

            # If scaling, store the scaled region dimensions
            if self.scale_to_mpp is not None:
                region_data['dimensions'] = (
                    convert_pixel_mpp(
                        region_data['dimensions'][0], region_data['mpp'][0],
                        self.scale_to_mpp, round_int=True),
                    convert_pixel_mpp(
                        region_data['dimensions'][1], region_data['mpp'][0],
                        self.scale_to_mpp, round_int=True))

            # Convert dimensions to a numpy array (for proper batching)
            region_data['dimensions'] = np.asarray(region_data['dimensions'])

            # Turn into a list so data can be sampled multiple times
            region_data_list = [region_data]

            # Create any tiles (if necessary)
            if self.tile_size is not None:
                region_data_list = self.tile_region_data_list(region_data_list)

            # Append the data to the region data store
            for region_data in region_data_list:
                for _ in range(self.samples_per_region):
                    all_region_data.append(region_data)

        return all_region_data, num_unique_images

    def tile_region_data_list(self, region_data_list: List) -> List:
        """Creates tiles from a list of region data, generating more regions as necessary.

        When tiling, the region_data is duplicated N times (per-tile), and the TILE_COORDS key
        added to the region_data.

        Args:
            region_data_list: A list of region data dictionaries

        Returns:
            A list of region data dictionaries, but with tile coordinates set as required.
        """
        tiled_region_data = []
        for region_data in region_data_list:
            # Extract coordinates used to generate tiles
            coords_to_tile = [0, 0, *region_data['dimensions']]

            # Generate the tiles (returns a list of numpy arrays)
            output_size = calculate_cropped_size(self.tile_size, self.output_crop_margin)
            all_tiles = generate_tiles(coords_to_tile, self.tile_size, output_size)

            # Append each tile to the tiled_region_data list
            for tile in all_tiles:
                # Create a copy of the original region data and set the TILE_COORDS key
                current_region_data = copy.deepcopy(region_data)
                current_region_data[TILE_COORDS] = tile
                tiled_region_data.append(current_region_data)

        return tiled_region_data

    def __len__(self) -> int:
        """Returns the length of the dataset.

        The length of the dataset is the total number of regions.

        Returns:
            The length of the dataset.
        """
        return len(self.all_region_data)

    def get_output_coordinates(self, input_region, region_data, offset_to_tile=True):
        """Given some region data, determines the output coordinates used for that region

        NOTE: By default, the coordinates will be relative to the input region (i.e. (0, 0) is the
            top-left of the input region)

        Args:
            input_region (ndarray/Tensor): The input region following all transforms. If tensor,
                expected ordering is CHW, if ndarray, expected ordering is HWC
            region_data (dict): Dictionary of data relating to the region being loaded
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
        if self.output_crop_margin is not None:
            output_height, output_width = \
                calculate_cropped_size((input_height, input_width), self.output_crop_margin)
            # Determine top/left coordinate of output
            output_x1 = int(round((input_width - output_width) / 2))
            output_y1 = int(round((input_height - output_height) / 2))

            # Set coords as x1, y1, x1 + width, y1 + height
            output_coords = [output_x1, output_y1,
                             output_x1 + output_width, output_y1 + output_height]

        # Convert output coordinates to a numpy array
        output_coords = np.asarray(output_coords)

        # Shift the output coords based on TILE_COORDS if required
        if offset_to_tile and TILE_COORDS in region_data:
            # Extract the tile coordinates
            tile_coords = region_data[TILE_COORDS]

            # Shift the output coords based on tile coords
            output_coords[[0, 2]] += tile_coords[0]
            output_coords[[1, 3]] += tile_coords[1]

        return output_coords

    def get_region_data_for_input_path(self, input_path):
        """Returns the first found region data with input path matching the given argument

        Args:
            input_path (str): Path of the input to search for

        Returns:
            (dict): Region data (for one of the regions) associated to that input path
        """
        try:
            region_data = next(dat for dat in self.all_region_data
                               if dat['image_path'] == input_path)
        except StopIteration:
            raise RuntimeError(f'Could not find data corresponding to input path {input_path}')
        return region_data
