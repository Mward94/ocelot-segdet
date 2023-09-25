from typing import Tuple, Optional, Dict, Any, Sequence, Union, List

import cv2
import numpy as np
import torch
from numpy.typing import NDArray

from util import macenko


# ##################################################################################################
#                                   PyTorch Helper Functions
# ##################################################################################################
def get_default_device() -> torch.device:
    """Gets the default device string.

    This is ``"cuda"`` if CUDA is available, otherwise ``"cpu"``.

    Returns:
        String representing the default device.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def move_data_to_device(data, device):
    """Moves data to a given device

    If <data> is a list/tuple, each element in <data> is individually moved to the device, and a
    list of data returned. Otherwise, <data> is moved to the device

    Args:
        data (Tensor/list/tuple): Data to move to the given device
        device (Device): Device to move data to

    Returns:
        data (Tensor/list): Data moved to the given device
    """
    # If any data is a tuple/list, put each element onto the device individually
    if isinstance(data, (tuple, list)):
        return [move_data_to_device(dat, device) for dat in data]
    if not isinstance(data, torch.Tensor):
        return data
    return data.to(device)


# ##################################################################################################
#                                   Geometry Helper Functions
# ##################################################################################################
def calculate_cropped_size(
        input_size: Tuple[int, int],
        crop_margin: Optional[int],
) -> Tuple[int, int]:
    """Calculate the output size given an input size and crop margin.

    Args:
        input_size: The size of the box before cropping.
        crop_margin: The amount to crop from each side of the box.

    Returns:
        The size of the box after cropping.
    """
    if crop_margin is None:
        return input_size
    output_size = (input_size[0] - crop_margin * 2, input_size[1] - crop_margin * 2)
    if any(x < 0 for x in output_size):
        raise ValueError(f'Crop margin {crop_margin} is too large for the input size {input_size}')
    return output_size


def convert_pixel_mpp(pixel_coordinate, original_mpp, new_mpp, round_int=False):
    """Converts a pixel captured at a given microns-per-pixel to another microns-per-pixel

    The physical measurement in microns is identical, but depending on the microns-per-pixel, the
    observed pixel coordinate will change

    This converts a pixel captured at an observed MPP to another MPP

    Args:
        pixel_coordinate (int/np.ndarray): The pixel to change observed MPP of
        original_mpp (float): The microns-per-pixel the pixel coordinate was observed in
        new_mpp (float): The microns-per-pixel the pixel coordinate should be observed in
        round_int (bool): If True, result will be rounded and converted to int

    Returns:
        (float/int): The pixel coordinate at the new MPP resolution
    """
    res = pixel_coordinate * (original_mpp / new_mpp)
    if round_int:
        res = int(round(res))
    return res


def convert_dimensions_to_mpp(dimensions: Sequence[float], from_mpp: Union[float, Sequence[float]],
                              to_mpp: Union[float, Sequence[float]], round_int=False,
                              validate_round=False):
    """Converts an (x, y) tuple from a given MPP to another MPP.

    Args:
        dimensions: The (x, y) dimensions to convert to a different MPP.
        from_mpp: The MPP that the dimensions are observed in.
        to_mpp: The MPP to convert dimensions to.
        round_int: Whether dimensions should be rounded to integers.
        validate_round: If rounding, this will compare the differences in rounding vs. not
            rounding, and raise a ValueError if any discrepancies are found.
    """
    if not isinstance(from_mpp, (list, tuple)):
        from_mpp = [from_mpp, from_mpp]
    if not isinstance(to_mpp, (list, tuple)):
        to_mpp = [to_mpp, to_mpp]

    converted_dims = np.asarray([
        convert_pixel_mpp(dimensions[0], from_mpp[0], to_mpp[0], round_int=round_int),
        convert_pixel_mpp(dimensions[1], from_mpp[1], to_mpp[1], round_int=round_int)])
    if round_int and validate_round:
        converted_dims_no_round = np.asarray([
            convert_pixel_mpp(dimensions[0], from_mpp[0], to_mpp[0], round_int=False),
            convert_pixel_mpp(dimensions[1], from_mpp[1], to_mpp[1], round_int=False)])

        if np.sum(np.abs(converted_dims - converted_dims_no_round)) != 0:
            raise ValueError(f'Rounding produced different results. Rounding: {converted_dims}. '
                             f'No rounding: {converted_dims_no_round}.')
    return converted_dims


def get_region_dimensions(region: Sequence[int]) -> Tuple[int, int]:
    """Given a region, returns the width & height of the region

    Args:
        region: The region given as [x1, y1, x2, y2]

    Returns:
        The width and height of the region
    """
    return int(region[2] - region[0]), int(region[3] - region[1])


def get_region_centre(region: Sequence[int]) -> Tuple[float, float]:
    """Given a region, returns the centre of the region

    Args:
        region: The region given as [x1, y1, x2, y2]

    Returns:
        The centre x and y position of the region
    """
    return (region[0] + region[2]) / 2, (region[1] + region[3]) / 2


def get_windowed_load_crop(image_shape, window):
    """Given a window, determines where to place a crop of the valid image data into that window

    Args:
        image_shape (2-tuple): The width, height of the image being loaded
        window (4-tuple): The x1, y1, x2, y2 coordinates of the window to be loaded

    Returns:
        (4-tuple): The x1, y1, x2, y2 coordinates of where the image should be positioned in an
            array with shape *window*
    """
    window_width, window_height = window[2] - window[0], window[3] - window[1]
    crop_x1 = np.clip(0 - window[0], a_min=0, a_max=window_width).item()
    crop_y1 = np.clip(0 - window[1], a_min=0, a_max=window_height).item()
    crop_x2 = window_width - np.clip(window[2] - image_shape[0], a_min=0, a_max=window_width).item()
    crop_y2 = window_height - np.clip(window[3] - image_shape[1], a_min=0,
                                      a_max=window_height).item()
    return max(0, crop_x1), max(0, crop_y1), max(0, crop_x2), max(0, crop_y2)


def scale_coords_to_mpp(
        coords: Union[NDArray[np.int32], NDArray[np.float32]],
        from_mpp: Union[int, float, Tuple[int, int], Tuple[float, float]],
        to_mpp: Union[int, float, Tuple[int, int], Tuple[float, float]], as_int: bool = False,
) -> Union[NDArray[np.int32], NDArray[np.float32]]:
    """Scales a collection of coordinates from one MPP to another.

    Scaling is performed as:
        x = x * (from_mpp [x] / to_mpp [x])
        y = y * (from_mpp [y] / to_mpp [y])

    Args:
        coords: The collection of coordinates to scale. This should have shape (N, 2).
        from_mpp: The MPP the coords are observed in (x, y). If given as a single value, assumes
            MPP is the same in both axes.
        to_mpp: The MPP the coords should be scaled to (x, y). If given as a single value, assumes
            MPP is the same in both axes.
        as_int: Whether to convert the coordinates back to integer before returning

    Returns:
        The coords scaled to to_mpp.
    """
    # Ensure correct dimensionality of coords (N, 2)
    assert coords.ndim == 2 and coords.shape[1] == 2, f'{coords.shape}'

    # Ensure from_mpp and to_mpp are tuples
    if isinstance(from_mpp, (int, float)):
        from_mpp = (from_mpp, from_mpp)
    if isinstance(to_mpp, (int, float)):
        to_mpp = (to_mpp, to_mpp)

    # Convert dtype to float to enable scaling. Ensure a copy of the original array is made
    scaled_coords = coords.astype(np.float32, copy=True)

    # Scale x/y coordinates by scale factors described in docstring
    scaled_coords[:, 0] *= (from_mpp[0] / to_mpp[0])
    scaled_coords[:, 1] *= (from_mpp[1] / to_mpp[1])

    # Return the coords, converting back to integers if required
    if as_int:
        scaled_coords = scaled_coords.astype(np.int32)
    return scaled_coords


# ##################################################################################################
#                                       Image Helpers
# ##################################################################################################
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


# ##################################################################################################
#                                       Macenko Normalisation
# ##################################################################################################
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


# ##################################################################################################
#                                   Cell-Tissue Patch Functions
# ##################################################################################################
def get_wsi_mpp(meta_pair: Dict[str, Any]) -> Tuple[float, float]:
    """Returns the MPP of the WSI. Returned in form: (X, Y)
    """
    return meta_pair['mpp_x'], meta_pair['mpp_y']


def get_region_mpp(meta_pair: Dict[str, Any], annot_type: str) -> Tuple[float, float]:
    """Returns the MPP of the cell/tissue patch. Returned in form: (X, Y).
    """
    if annot_type not in {'cell', 'tissue'}:
        raise ValueError(f'Must specify cell/tissue. {annot_type} invalid.')
    return meta_pair[annot_type]['resized_mpp_x'], meta_pair[annot_type]['resized_mpp_y']


def get_region_wsi_coordinates(
        meta_pair: Dict[str, Any], annot_type: str,
        mpp: Optional[Union[float, Sequence[float]]] = None,
) -> Tuple[int, int, int, int]:
    """Gets the Cell/Tissue coordinates at a given MPP.

    If MPP is None, returns at the WSI MPP.

    Returned in form: x1, y1, x2, y2
    """
    if annot_type not in {'cell', 'tissue'}:
        raise ValueError(f'Must specify cell/tissue. {annot_type} invalid.')

    # Get the coordinates of the cell crop at the WSI MPP
    x1, y1 = meta_pair[annot_type]['x_start'], meta_pair[annot_type]['y_start']
    x2, y2 = meta_pair[annot_type]['x_end'], meta_pair[annot_type]['y_end']
    if mpp is None:
        return x1, y1, x2, y2

    if isinstance(mpp, (int, float)):
        mpp = (mpp, mpp)

    # If scaling to other MPP, determine what MPP it is observed in (WSI MPP)
    original_mpp = get_wsi_mpp(meta_pair)

    # Scale coordinates to new MPP
    original_width, original_height = x2 - x1, y2 - y1
    x1 = convert_pixel_mpp(x1, original_mpp[0], mpp[0], round_int=True)
    y1 = convert_pixel_mpp(y1, original_mpp[1], mpp[1], round_int=True)
    width = convert_pixel_mpp(original_width, original_mpp[0], mpp[0], round_int=True)
    height = convert_pixel_mpp(original_height, original_mpp[1], mpp[1], round_int=True)
    x2 = x1 + width
    y2 = y1 + height
    return x1, y1, x2, y2


def cell_scale_crop_in_tissue_at_cell_mpp(
        meta_pair: Dict[str, Any],
        tissue_mpp: Optional[Union[float, Tuple[float, float]]] = None,
        cell_mpp: Optional[Union[float, Tuple[float, float]]] = None,
) -> Tuple[Tuple[float, float], List[int]]:
    """Gets the coordinates of the crop to take in the tissue region at the cell MPP.

    tissue_mpp is the MPP that the tissue data exists in (used for scale information).

    cell_mpp is the MPP the desired cell data should be at.

    If either tissue_mpp or cell_mpp not given, uses what is stored in the file
    """
    # Get MPP of cell/tissue/WSI
    if cell_mpp is None:
        cell_mpp = get_region_mpp(meta_pair, 'cell')
    else:
        if isinstance(cell_mpp, (int, float)):
            cell_mpp = (cell_mpp, cell_mpp)
    if tissue_mpp is None:
        tissue_mpp = get_region_mpp(meta_pair, 'tissue')
    else:
        if isinstance(tissue_mpp, (int, float)):
            tissue_mpp = (tissue_mpp, tissue_mpp)
    wsi_mpp = get_wsi_mpp(meta_pair)

    # Determine the scaling between mask at tissue MPP vs. cell MPP (make larger)
    scale_factor_x, scale_factor_y = tissue_mpp[0] / cell_mpp[0], tissue_mpp[1] / cell_mpp[1]

    # Get the coordinates of the tissue area at the WSI MPP
    tissue_wsi_coords = get_region_wsi_coordinates(meta_pair, annot_type='tissue', mpp=None)

    # Get scale factor for WSI-MPP to cell MPP
    wsi_cell_sf_x, wsi_cell_sf_y = wsi_mpp[0] / cell_mpp[0], wsi_mpp[1] / cell_mpp[1]

    # Scale tissue x1, y1 from WSI MPP to cell MPP
    tissue_cell_x1, tissue_cell_y1 = tissue_wsi_coords[0] * wsi_cell_sf_x, tissue_wsi_coords[1] * wsi_cell_sf_y

    # Extract the coordinates of the cell box (at cell MPP)
    cell_coords = get_region_wsi_coordinates(meta_pair, annot_type='cell', mpp=cell_mpp)

    # Set the crop coordinates relative to tissue_cell_x1/y1
    crop_coords = [int(round(cell_coords[0] - tissue_cell_x1)),
                   int(round(cell_coords[1] - tissue_cell_y1)),
                   int(round(cell_coords[2] - tissue_cell_x1)),
                   int(round(cell_coords[3] - tissue_cell_y1))]

    return (scale_factor_x, scale_factor_y), crop_coords
