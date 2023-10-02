"""Utilities relating to dealing with image data.
"""
import os
import warnings
from typing import Optional, Sequence, Union, Tuple, List, Dict, Any

import cv2
import numpy as np
import rasterio
from numpy.typing import NDArray
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from util import macenko
from util.helpers import get_region_dimensions, get_windowed_load_crop


DEFAULT_TIF_PROFILE = {
    'driver': 'GTiff',
    'dtype': np.uint8,
    'count': None,
    'compress': 'DEFLATE',
    'ZLEVEL': 1,
    'PREDICTOR': 2,
    'interleave': 'pixel',
    'width': None,
    'height': None,
    'tiled': True,
    'blockxsize': 64,
    'blockysize': 64,
    'BIGTIFF': 'YES',  # Use BigTIFF, so we can handle files larger than 4GB.
    'transform': rasterio.Affine(1, 0, 0, 0, 1, 0),
}


def write_tif_rasterio(
    filepath: str,
    image: np.ndarray,
    profile: dict,
    overwrite: bool = False,
    pyramid_factors: Optional[Sequence[int]] = None,
    resampling: Optional[Resampling] = None,
):
    """Writes an image as a TIF file using rasterio

    Args:
        filepath: Filepath to write data to
        image: Image data to write. If colour, should be RGB, HWC axis ordering.
            If grayscale, should have HW axis ordering
        profile: The profile to use for storing to disk with rasterio
        overwrite: Whether to overwrite the existing file if it exists
        pyramid_factors: Resampling factors for the image pyramid. If ``None``, no image pyramid
            will be saved.
        resampling: The resampling algorithm to use when generating the image pyramid.
    """
    # Check if the file already exists
    if not overwrite and os.path.isfile(filepath):
        raise FileExistsError(f'Error. File \'{filepath}\' already exists. Set the overwrite flag '
                              f'to overwrite the file')

    # Write the data
    n_channels = 1 if image.ndim == 2 else image.shape[2]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(filepath, 'w+', **profile) as out_tif:
            if n_channels == 1:
                out_tif.write(image, 1)
            else:
                for i in range(n_channels):
                    out_tif.write(image[..., i], i + 1)
            if pyramid_factors is not None and len(pyramid_factors) >= 1:
                out_tif.build_overviews(pyramid_factors, resampling)
                out_tif.update_tags(ns='rio_overview', resampling=resampling.name)


def load_tif_rasterio(
    filepath: str,
    window: Optional[Sequence[int]] = None,
    out_size: Optional[Sequence[int]] = None,
    resampling: Optional[Union[str, Resampling]] = None,
    fill_value: int = 0,
    dtype=np.uint8,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int, int]]]:
    """Loads a TIF image using rasterio.

    The image will be returned as a HWC numpy array (or a HW array if there is only one channel).

    Args:
        filepath: Path to the TIF file to load.
        window: The x1, y1, x2, y2 coordinates of the rectangular sub-image to load from the full
            image. Specifying a window in this way enables image loading optimisations. If part of
            the window lies outside the full image bounds, padding will be applied.
        out_size: The width, height to resize the loaded image data to. Specifying an output size
            in this way enables image loading optimisations when the TIF file contains an image
            pyramid.
        resampling: The resampling algorithm to use when resizing. If None, the resampling algorithm
            will be inferred from the TIF file or fall back to average if that's not possible.
        fill_value: The fill value to use for out of bounds pixels when reading the image.
        dtype: The datatype to load data as.

    Returns:
        The loaded image data and the x1, y1, x2, y2 coordinates of the rectangular region inside
        the loaded image where valid data exists (or ``None`` if it is all valid).
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f'Cannot load file: {filepath} - Does not exist.')

    if window is None:
        window_size = None
    else:
        window_size = get_region_dimensions(window)

    crop_window = None
    with rasterio.open(filepath) as in_tif:
        in_size = (in_tif.width, in_tif.height)

        if out_size is None:
            if window is None:
                out_size = in_size
            else:
                out_size = window_size

        if resampling is None:
            resampling = Resampling[in_tif.tags(ns='rio_overview').get('resampling', 'average')]
        elif isinstance(resampling, str):
            resampling = Resampling[resampling]

        # Create an empty array to load data into
        image_data = np.full((in_tif.count, out_size[1], out_size[0]), fill_value=fill_value,
                             dtype=dtype)

        if window is None:
            image_data_valid = image_data
        else:
            # Get the region where data should be loaded into
            cx1, cy1, cx2, cy2 = get_windowed_load_crop(in_size, window)

            # Scale the crop window using out_size
            scale_x = out_size[0] / window_size[0]
            scale_y = out_size[1] / window_size[1]
            cx1 = int(cx1 * scale_x)
            cy1 = int(cy1 * scale_y)
            cx2 = int(cx2 * scale_x)
            cy2 = int(cy2 * scale_y)
            image_data_valid = image_data[..., cy1:cy2, cx1:cx2]
            crop_window = (cx1, cy1, cx2, cy2)

        rio_window = None
        if window is not None:
            rio_window = from_bounds(
                left=window[0], top=window[1], right=window[2], bottom=window[3],
                transform=in_tif.transform
            )

        # Load data into that region (everything else becomes padding)
        in_tif.read(out=image_data_valid, window=rio_window, resampling=resampling)

    # Switch from CHW to HWC axis order
    image_data = np.transpose(image_data, (1, 2, 0))

    # Remove the channel axis if there is only one channel
    if image_data.shape[-1] == 1:
        image_data = np.squeeze(image_data, -1)

    return image_data, crop_window


def write_image(filepath: str, image: NDArray[np.uint8], overwrite: bool = False):
    """Writes an image to file.

    Args:
        filepath: The filepath to write the data to.
        image: The image data to write. Should be RGB, HWC axis ordering.
        overwrite: Whether to overwrite the existing file if it exists.

    Raises:
        FileExistsError: If `filepath` exists and `overwrite` is False.
    """
    # Check if the file already exists
    if not overwrite and os.path.isfile(filepath):
        raise FileExistsError(f'File \'{filepath}\' already exists. Set the overwrite flag '
                              f'to overwrite the file')

    # Write the data
    cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def load_image(filepath: str) -> NDArray[np.uint8]:
    """Loads a RGB image into a numpy array

    Args:
        filepath: Filepath to the image to be loaded

    Returns:
        The loaded RGB image, with channel ordering HWC
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f'Cannot load file: {filepath} - Does not exist.')
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)


def get_tissue_mask(
        image: NDArray[np.uint8],
        luminosity_threshold: float = 0.8,
        mask_out_black: bool = False,
) -> NDArray[bool]:
    """Computes a foreground tissue mask.

    This is done by converting the image to the LAB colour space, then thresholding the L
    (luminosity) channel.

    Args:
        image: Image to compute foreground mask of.
        luminosity_threshold: The luminosity threshold [0, 1] to mask pixels in/out. Values < this
            threshold are retained.
        mask_out_black: Whether pure black pixels should also be masked out

    Returns:
        The boolean mask.
    """
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    if not mask_out_black:
        return lab_image[..., 0] < (luminosity_threshold * 255)
    return np.logical_and(lab_image[..., 0] < (luminosity_threshold * 255), lab_image[..., 0] > 0)


def crop_image(image: NDArray, box: Union[Tuple[int, int, int, int], List[int]],
               allow_pad: bool = False, pad_fill_value: int = 0) -> NDArray:
    """Crops an image given a bounding box region

    The box should be in the form (x1, y1, x2, y2), as the top-left and bottom-right coordinates

    Assumes image stored in HWC format (or HW if a grayscale image)

    Args:
        image: Image to be cropped
        box: Bounding box coordinates of region to crop
        allow_pad: Whether padding is allowed (applied if the box has negative coordinates, or
            coordinates beyond the image dimensions).
        pad_fill_value: If padding applied, the pad colour.

    Returns:
        The cropped image.
    """
    # If padding not allowed, crop directly from the image
    if not allow_pad:
        return image[int(box[1]):int(box[3]), int(box[0]):int(box[2]), ...]

    # Convert box to numpy array (makes a copy to not modify original box)
    box = np.array(box)

    # Get the size of the crop to take
    crop_size = get_region_dimensions(box)

    # H, W, C
    if image.ndim == 2:
        image_data = np.full((crop_size[1], crop_size[0]), fill_value=pad_fill_value,
                             dtype=image.dtype)
    else:
        image_data = np.full((crop_size[1], crop_size[0], image.shape[2]),
                             fill_value=pad_fill_value,
                             dtype=image.dtype)

    # Get the region where data should be loaded into
    cx1, cy1, cx2, cy2 = get_windowed_load_crop((image.shape[1], image.shape[0]), box)

    # Bound box crop to image
    box[[0, 2]] = np.clip(box[[0, 2]], a_min=0, a_max=image.shape[1])
    box[[1, 3]] = np.clip(box[[1, 3]], a_min=0, a_max=image.shape[0])

    # Load image data
    image_data[cy1:cy2, cx1:cx2, ...] = image[box[1]:box[3], box[0]:box[2], ...]

    return image_data


def precompute_macenko_params(image: NDArray[np.uint8]) -> Dict[str, Any]:
    """Precomputes the Macenko normalisation parameters if required.

    Normalisation will be computed on the image as-is (at it's given resolution).

    Tissue masking will be applied
    """
    # Load the tissue mask
    tissue_mask = get_tissue_mask(image)

    # Mask the image
    image = image[tissue_mask]

    # Generate the Macenko parameters
    macenko_params = macenko.precompute_imagewide_normalisation_parameters(image)

    # Return the computed Macenko parameters
    return macenko_params


def overlay_images(original, data_to_overlay, alpha, beta=None, gamma=0):
    """Overlays an image ontop of another one

    Inspiration: https://www.pyimagesearch.com/2016/03/07/transparent-overlays-with-opencv/
    """
    if beta is None:
        beta = 1 - alpha
    return cv2.addWeighted(data_to_overlay, alpha, original, beta, gamma)
