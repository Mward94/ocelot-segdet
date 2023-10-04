"""This script loads and parses annotations for data in the OCELOT dataset.

As part of preprocessing, this will:
* Process all image data and store in a tiled TIF format
* Apply Macenko normalisation (if requested) to image data before storing
* Load and store the set of annotations per-case
* Store pickle files per-case that contain metadata about the images
"""
import argparse
import csv
import os
from typing import List, Dict

import numpy as np
from rasterio.enums import Resampling
from tqdm import tqdm

from util import macenko
from util.constants import TISSUE_CLASS_COLOURS, CELL_CLASS_COLOURS, CELL_CLASSES
from util.gcio import read_json
from util.helpers import (
    get_basename_from_filepath, get_recursive_directory_listing, create_directory, to_relpath,
    draw_points_on_image, write_pickle_data)
from util.image import (
    write_tif_rasterio, load_image, get_tissue_mask, DEFAULT_TIF_PROFILE, write_image,
    overlay_images)
from util.ocelot_parsing import get_region_mpp, TISSUE_LABEL_MAP, CELL_LABEL_MAP

TIF_PYRAMID_FACTORS = [2, 4, 8]


def parse_args():
    parser = argparse.ArgumentParser()

    # Input data directory
    parser.add_argument('--ocelot-directory', type=str,
                        help='Path to OCELOT data directory. Should contain annotations/ '
                             'directory, images/ directory, and metadata.json file.', required=True)

    # Which image types should me Macenko normalised ('cell'/'tissue')
    parser.add_argument('--macenko', type=str, nargs='+', choices=['cell', 'tissue'],
                        help='Which images to apply Macenko normalisation to.', default=['tissue'])

    # If samples of overlaid annotations should also be extracted
    parser.add_argument('--extract-overlays', action=argparse.BooleanOptionalAction,
                        help='Also extract images of annotations overlaid on the images.',
                        default=False)

    # Output directory for processed data
    parser.add_argument('--output-directory', type=str,
                        help='Path to directory to store processed data in.', required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    ocelot_dir = args.ocelot_directory
    out_dir = args.output_directory

    # Ensure directory is valid
    check_valid_ocelot_directory(ocelot_dir)

    # Find the set of images to be processed
    ocelot_id_list = get_ocelot_file_ids(os.path.join(ocelot_dir, 'annotations'))
    ocelot_id_list.sort()

    # Set up directory names to store processed data in
    image_directory = os.path.join(out_dir, 'images')           # Image data
    tissue_seg_directory = os.path.join(out_dir, 'tissue_seg')  # Tissue annotations
    cell_det_directory = os.path.join(out_dir, 'cells')         # Cell detections
    metadata_directory = os.path.join(out_dir, 'metadata')      # Metadata per-case
    overlays_directory = os.path.join(out_dir, 'overlays')      # Overlays (if generating)

    # Ensure all directories exist
    create_directory(out_dir)
    create_directory(image_directory)
    create_directory(tissue_seg_directory)
    create_directory(cell_det_directory)
    create_directory(metadata_directory)
    if args.extract_overlays:
        create_directory(overlays_directory)

    # Load the metadata.json file
    ocelot_metadata = read_json(os.path.join(ocelot_dir, 'metadata.json'))['sample_pairs']

    # Iterate through all IDs and preprocess their data
    progress = tqdm(ocelot_id_list)
    for im_id in progress:
        progress.set_description(f'Processing image pair with ID: {im_id}')

        # Extract the metadata for this case
        ocelot_case_metadata = ocelot_metadata[im_id]

        # Set paths to store processed image and annotation data
        tissue_image_path = os.path.join(image_directory, f'{im_id}_tissue.tif')
        tissue_gt_path = os.path.join(tissue_seg_directory, f'{im_id}.tif')
        cell_image_path = os.path.join(image_directory, f'{im_id}_cell.tif')
        metadata_path = os.path.join(metadata_directory, f'{im_id}.pkl')

        # Store representing information about the case
        case_info = {
            'id': im_id,
            'tissue': {     # MPP and paths to where processed data stored
                'mpp': get_region_mpp(ocelot_case_metadata, 'tissue'),
                'dimensions': [],
                'image_path': to_relpath(tissue_image_path, out_dir),
                'gt_path': to_relpath(tissue_gt_path, out_dir),
            },
            'cell': {       # MPP and paths to where processed data stored
                'mpp': get_region_mpp(ocelot_case_metadata, 'cell'),
                'dimensions': [],
                'image_path': to_relpath(cell_image_path, out_dir),
                'gt_path': {},  # class_name: path
            }
        }

        # Get reference to paths of the image and annotation data
        image_paths = get_image_paths(os.path.join(ocelot_dir, 'images'), im_id)
        annotation_paths = get_annotation_paths(os.path.join(ocelot_dir, 'annotations'), im_id)

        # ### Store 'tissue' image ###
        # Load the tissue image
        tissue_image = load_image(image_paths['tissue'])
        case_info['tissue']['dimensions'] = [tissue_image.shape[1], tissue_image.shape[0]]

        # Perform Macenko normalisation (if requested)
        if 'tissue' in args.macenko:
            tissue_mask = get_tissue_mask(tissue_image)
            tissue_image = macenko.normalise_he_image(tissue_image, mask=tissue_mask)

        # Write the tissue image
        tif_profile = DEFAULT_TIF_PROFILE.copy()
        tif_profile['count'] = tissue_image.shape[2]
        tif_profile['width'] = tissue_image.shape[1]
        tif_profile['height'] = tissue_image.shape[0]
        write_tif_rasterio(tissue_image_path, tissue_image, tif_profile, overwrite=True,
                           pyramid_factors=TIF_PYRAMID_FACTORS, resampling=Resampling.average)

        # ### Store 'tissue' annotations ###
        # Load the tissue segmentation mask (as HWC -- (1024, 1024, 3))
        seg_mask = load_image(annotation_paths['tissue'])

        # Ensure all 3 channels match, then extract a single channel
        assert (np.array_equal(seg_mask[..., 0], seg_mask[..., 1]) and
                np.array_equal(seg_mask[..., 0], seg_mask[..., 2]))
        seg_mask = seg_mask[..., 0]

        # Remap to match class order to be predicted
        int_seg_mask = np.ones_like(seg_mask, dtype=np.uint8) * 255
        for orig_cls, new_cls in TISSUE_LABEL_MAP.items():
            int_seg_mask[seg_mask == orig_cls] = new_cls

        # Ensure only 'values' present in the integer segmentation mask
        valid_values_set = set(TISSUE_LABEL_MAP.values())
        mask_values_set = set(np.unique(int_seg_mask))
        assert mask_values_set.issubset(valid_values_set)

        # Write the tissue annotations
        tif_profile = DEFAULT_TIF_PROFILE.copy()
        tif_profile['count'] = 1
        tif_profile['width'] = int_seg_mask.shape[1]
        tif_profile['height'] = int_seg_mask.shape[0]
        write_tif_rasterio(tissue_gt_path, int_seg_mask, tif_profile, overwrite=True,
                           pyramid_factors=TIF_PYRAMID_FACTORS, resampling=Resampling.nearest)

        # ### Store 'tissue' annotations overlaid on image (if requested) ###
        if args.extract_overlays:
            tissue_overlay_path = os.path.join(overlays_directory, f'{im_id}_tissue.jpg')
            # Generate RGB mask
            tissue_overlay = np.zeros_like(tissue_image)
            for cls_idx, cls_colour in enumerate(TISSUE_CLASS_COLOURS):
                tissue_overlay[int_seg_mask == cls_idx] = cls_colour
            # Blend with original image
            tissue_overlay = overlay_images(tissue_image, tissue_overlay, 0.25)
            write_image(tissue_overlay_path, tissue_overlay, overwrite=True)

        # ### Store 'cell' image ###
        # Load the cell image
        cell_image = load_image(image_paths['cell'])
        case_info['cell']['dimensions'] = [cell_image.shape[1], cell_image.shape[0]]

        # Perform Macenko normalisation (if requested)
        if 'cell' in args.macenko:
            cell_mask = get_tissue_mask(cell_image)
            cell_image = macenko.normalise_he_image(cell_image, mask=cell_mask)

        # Write the cell image
        tif_profile = DEFAULT_TIF_PROFILE.copy()
        tif_profile['count'] = cell_image.shape[2]
        tif_profile['width'] = cell_image.shape[1]
        tif_profile['height'] = cell_image.shape[0]
        write_tif_rasterio(cell_image_path, cell_image, tif_profile, overwrite=True,
                           pyramid_factors=TIF_PYRAMID_FACTORS, resampling=Resampling.average)

        # ### Store 'cell' annotations ###
        cell_annotations = {}       # Mapping from class name to [(x, y), ...] coordinates

        # Load the cell annotations file
        with open(annotation_paths['cell'], 'r') as csv_file:
            reader = csv.DictReader(csv_file, fieldnames=['x', 'y', 'label'])
            cell_annot_data = [dict(row) for row in reader]

        # Data stored as: x, y (region-relative), label
        for dat in cell_annot_data:
            x, y = int(dat['x']), int(dat['y'])
            classname = CELL_CLASSES[CELL_LABEL_MAP[dat['label']]]
            if classname not in cell_annotations:
                cell_annotations[classname] = []
            cell_annotations[classname].append([x, y])

        # Convert to numpy array and write data
        for name in cell_annotations.keys():
            cell_annotations[name] = np.asarray(cell_annotations[name])

            # Create folder based on class name
            cell_gt_directory = os.path.join(cell_det_directory, name)
            create_directory(cell_gt_directory)
            cell_gt_path = os.path.join(cell_gt_directory, f'{im_id}.npy')
            case_info['cell']['gt_path'][name] = to_relpath(cell_gt_path, out_dir)

            # Write to numpy array
            np.save(cell_gt_path, cell_annotations[name])

        # ### Store 'cell' annotations overlaid on image (if requested) ###
        if args.extract_overlays:
            cell_overlay_path = os.path.join(overlays_directory, f'{im_id}_cell.jpg')

            # Overlay points
            cell_overlay = cell_image.copy()
            for name in cell_annotations.keys():
                colour = CELL_CLASS_COLOURS[CELL_CLASSES.index(name)]
                draw_points_on_image(cell_overlay, cell_annotations[name], colour,
                                     radius=5, thickness=2, inplace=True)
            write_image(cell_overlay_path, cell_overlay, overwrite=True)

        # ### Store 'metadata' file ###
        write_pickle_data(metadata_path, case_info, overwrite=True)


def check_valid_ocelot_directory(directory: str):
    """Checks the directory contains annotations/, images/, and metadata.json
    """
    if not os.path.isdir(os.path.join(directory, 'annotations')):
        raise ValueError(f'Could not find annotations directory in \'{directory}\'')
    if not os.path.isdir(os.path.join(directory, 'images')):
        raise ValueError(f'Could not find images directory in \'{directory}\'')
    if not os.path.isfile(os.path.join(directory, 'metadata.json')):
        raise ValueError(f'Could not find metadata file in \'{directory}\'')


def get_ocelot_file_ids(annotation_directory: str) -> List[str]:
    """Gets all file IDs that exist within the annotation directory for the Ocelot dataset.

    Args:
        annotation_directory: Path to the directory containing annotation data.

    Returns:
        List of patient IDs that were found.
    """
    # This searches for files based on the tissue annotation (.png image files)
    return [get_basename_from_filepath(path) for path in
            get_recursive_directory_listing(
                annotation_directory, search_depth=-1, extension_whitelist='.png')]


def get_image_paths(image_directory: str, case_id: str) -> Dict[str, str]:
    """Gets the paths to the tissue and cell images for a case in the Ocelot dataset.

    Args:
        image_directory: Path to the directory containing image data.
        case_id: ID of the case to find path for.

    Returns:
        Mapping of 'tissue' and 'cell' to path to that image.
    """
    # Should find both for 'cell' and 'tissue'
    paths = get_recursive_directory_listing(
        image_directory, search_depth=-1, extension_whitelist='.jpg',
        filename_start_filter=case_id)
    assert len(paths) == 2

    return {determine_ocelot_path_type(p): p for p in paths}


def get_annotation_paths(annotation_directory: str, case_id: str) -> Dict[str, str]:
    """Gets the annotation paths for cases belonging to the Ocelot dataset.

    Args:
        annotation_directory: Path to the directory containing annotation data.
        case_id: ID of the case to find path for.

    Returns:
        Mapping of 'tissue' and 'cell' to path to that annotation.
    """
    # Should find both for 'cell' and 'tissue'
    paths = get_recursive_directory_listing(
        annotation_directory, search_depth=-1, extension_whitelist=('.csv', '.png'),
        filename_start_filter=case_id)
    assert len(paths) == 2

    return {determine_ocelot_path_type(p): p for p in paths}


def determine_ocelot_path_type(path):
    """Determines if a path corresponds to a 'tissue' or 'cell' annotation/file.
    """
    # Determine if this corresponds to the 'cell' or 'tissue' image
    annot_type = 'cell' if os.path.dirname(path).endswith('cell') else 'tissue' \
        if os.path.dirname(path).endswith('tissue') else None
    if annot_type is None:
        raise RuntimeError(f'Could not determine if cell or tissue data from {path}.')
    return annot_type


if __name__ == '__main__':
    main()
