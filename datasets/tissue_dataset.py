"""Torch dataset used for loading tissue images with segmentation ground truth.
"""
import copy
import os
import pickle
from typing import Optional, Union, Tuple, List

import albumentations as A
import numpy as np
from albumentations import ToFloat
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from util.constants import TISSUE_CLASSES
from util.helpers import convert_pixel_mpp, calculate_cropped_size
from util.tiling import generate_tiles


# Region data key used to show we are tiling.
TILE_COORDS = '_TILE_COORDINATES'


class TissueDataset(Dataset):
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
        standard_transforms = [ToFloat(), ToTensorV2()]
        if transforms is None:
            self.transform = A.Compose(standard_transforms)
        else:
            self.transform = A.Compose([*transforms, *standard_transforms])

        # Store the name of the class to use as 'background'
        if pad_class_name is not None and pad_class_name not in TISSUE_CLASSES:
            raise ValueError(f'Expected pad_class_name={pad_class_name} to be in '
                             f'class_list={TISSUE_CLASSES}.')
        self.pad_class_name = pad_class_name

        # Set up store of annotation metadata
        self.all_region_data = self._load_from_split()

        for tx in self.transform:
            print(tx.__class__.__name__)

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

        # Create a store of dataset -> data path (for efficiency)
        dataset_data_store = {}

        # Create a datastore by region
        all_region_data = []

        # Populate the datastore
        for metadata in all_metadata:
            region_data = metadata['tissue']
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

        return all_region_data

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

    # TODO: Implement __getitem__
