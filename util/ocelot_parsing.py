"""Utility functions specific to the OCELOT dataset.
"""
from typing import Dict, Tuple, Any, Optional, Union, Sequence, List

from util.constants import TISSUE_CLASSES, CELL_CLASSES
from util.helpers import convert_pixel_mpp


# Mapping from tissue labels (stored in mask) to ones to be predicted
# In mask: 1=BG tissue, 2=Cancer area, 255=Unknown
TISSUE_LABEL_MAP = {
    1: TISSUE_CLASSES.index('Other'),
    2: TISSUE_CLASSES.index('Cancer'),
    255: TISSUE_CLASSES.index('Background'),
}

# Mapping from cell indices (stored in CSVs) to ones to be predicted
# In CSVs: 1=BG cell, 2=Tumour cell
CELL_LABEL_MAP = {
    '1': CELL_CLASSES.index('Background_Cell'),
    '2': CELL_CLASSES.index('Tumour_Cell'),
}


def map_tissue_classes():
    """Maps from tissue class indices stored on disk to tissue classes to predict.
    """
    return {}


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
