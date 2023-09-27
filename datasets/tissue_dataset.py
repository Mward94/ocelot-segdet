"""Torch dataset used for loading tissue images with segmentation ground truth.
"""
import os
import pickle
from typing import Optional, Union, Tuple, List

import albumentations as A
from albumentations import ToFloat
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from util.constants import TISSUE_CLASSES


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
        # TODO: FROM HERE
        for metadata in all_metadata:
            print(metadata)
            assert False
            # TODO: Just the tissue stuff
            # This represents coords of image relative to WSI (regardless of any scaling)
            if 'original_coordinates' in region_data:
                original_coordinates = region_data['original_coordinates']
            else:
                original_coordinates = region_data['coordinates']
            region_data['original_coordinates'] = np.asarray(original_coordinates)

            # Store the dimensions of the whole region
            region_data['region_dimensions'] = get_region_dimensions(
                region_data.pop('coordinates'))
            if self.scale_to_mpp is not None:
                # If scaling, store the scaled region dimensions
                region_data['region_dimensions'] = (
                    convert_pixel_mpp(
                        region_data['region_dimensions'][0], region_data['scale_info']['mpp_x'],
                        self.scale_to_mpp, round_int=True),
                    convert_pixel_mpp(
                        region_data['region_dimensions'][1], region_data['scale_info']['mpp_y'],
                        self.scale_to_mpp, round_int=True))

            # Convert region_dimensions to a numpy array (for proper batching)
            region_data['region_dimensions'] = np.asarray(region_data['region_dimensions'])

            # Filter/create any additional/modified regions
            region_data_list = [region_data]

            # Create any tiles (if necessary)
            # TODO
            if self.tile_size is not None:
                region_data_list = self.tile_region_data_list(region_data_list)
            # END TODO

            # Append the data to the region data store
            for region_data in region_data_list:
                for _ in range(self.samples_per_region):
                    all_region_data.append(region_data)

        return all_region_data
