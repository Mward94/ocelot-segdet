import os
import pickle
from typing import Tuple, Optional, Sequence, Union, List, Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray


# ##################################################################################################
#                                       File Helper Functions
# ##################################################################################################
def get_basename_from_filepath(filepath: str) -> str:
    """Extracts the basename of a file from the filepath.

    Args:
        filepath: File path.

    Returns:
        Basename of the file, excluding the file extension.
    """
    return os.path.splitext(os.path.basename(filepath))[0]


def get_extension_from_filepath(filepath: str) -> str:
    """Extracts the extension of a file from the filepath.

    Args:
        filepath: File path.

    Returns:
        Extension of the file, or an empty string if there is no extension.
    """
    return os.path.splitext(filepath)[1]


def create_directory(directory: str):
    """Creates a new directory, regardless of whether it exists or not.

    Directories will be created recursively.

    If `directory` is an empty string, or the directory already exists, this function is a no-op.

    Args:
        directory: Directory to create.
    """
    if directory:
        os.makedirs(directory, exist_ok=True)


def replace_extension(filepath: str, new_extension: str) -> str:
    """Replaces the extension of a filepath.

    Args:
        filepath: File path to replace the extension of.
        new_extension: New extension to replace the old one (MUST INCLUDE THE PERIOD).

    Returns:
        File path with the replaced extension.
    """
    directory = os.path.dirname(filepath)
    basename = get_basename_from_filepath(filepath)
    return os.path.join(directory, basename + new_extension)


def get_nested_directory_level(base_directory: str, nested_directory: str) -> int:
    """Gets the nested level of a directory relative to a base directory.

    The nested level is 0 if the directories are identical, otherwise is the number of
    subdirectories that nested_directory is relative to base_directory.

    Args:
        base_directory: Path to the base directory
        nested_directory: Path to the nested directory

    Returns:
        The nested directory level of the nested directory relative to the base directory.

    Examples:
        >>> get_nested_directory_level('/home/john', '/home/john/.config/myconf.yml')
        2
    """
    base_directory = os.path.normpath(base_directory)
    nested_directory = os.path.normpath(nested_directory)

    if not nested_directory.startswith(base_directory):
        raise OSError(f'Nested directory: \'{nested_directory}\' is not located within base '
                      f'directory: \'{base_directory}\'')

    # Remove the base directory part from the nested directory, then count the number of separators
    difference = nested_directory[len(base_directory):]
    return difference.count(os.sep)


def get_recursive_directory_listing(
    directory: str,
    search_depth: int = -1,
    extension_whitelist: Optional[Union[str, Sequence[str]]] = None,
    filename_start_filter: Optional[str] = None,
) -> List[str]:
    """Recursively scans a directory and returns a list of all files found.

    Can also optionally blacklist or whitelist file extensions.

    Args:
        directory: Directory to search.
        search_depth: How deep directories should be searched. A search_depth of ``0`` only
            searches the current directory. A search_depth of ``-1`` searches all directories.
        extension_whitelist: A list of extensions to include.
        filename_start_filter: If specified, only files starting with this substring will
            be returned.

    Returns:
        List of all matching files.
    """
    # Set up the whitelist
    if extension_whitelist is not None:
        if not isinstance(extension_whitelist, (list, tuple)):
            extension_whitelist = [extension_whitelist]
    else:
        extension_whitelist = []

    # Handle a search depth of 0 (simple os.listdir())
    if search_depth == 0:
        if filename_start_filter is None:
            file_list = [os.path.join(directory, file) for file in os.listdir(directory)]
        else:
            file_list = [os.path.join(directory, file) for file in os.listdir(directory)
                         if file.startswith(filename_start_filter)]
        file_list = [file for file in file_list if not os.path.islink(file)]
        if extension_whitelist:
            file_list = [file for file in file_list
                         if get_extension_from_filepath(file) in extension_whitelist]
        return file_list

    # Track the number of files processed and matching files found so far
    file_list = []

    # Iterate through from the starting directory
    for dirpath, dirnames, filenames in os.walk(directory):
        # Ensure we don't search too deep
        if search_depth >= 0:
            directory_depth = get_nested_directory_level(directory, dirpath)
            if directory_depth > search_depth:
                continue

        # Iterate through files in the current directory. Track them if they match the filters
        for f in filenames:
            # Skip if filtering on filename_start
            if filename_start_filter is not None and not f.startswith(filename_start_filter):
                continue

            fp = os.path.join(dirpath, f)

            # Skip if a symbolic link
            if not os.path.islink(fp):
                # Skip if the extension is blacklisted, or isn't on the whitelist
                valid_file = True
                if extension_whitelist and get_extension_from_filepath(fp) not in extension_whitelist:
                    valid_file = False
                if valid_file:
                    file_list.append(fp)
    return file_list


def to_relpath(path, rel_to):
    path = os.path.relpath(path, rel_to)
    path = path.replace(os.sep, '/')
    return path


def write_pickle_data(filepath: str, data: Any, overwrite: bool = False):
    """Writes pickled data to file.

    Args:
        filepath: The file path to write the data to.
        data: Any picklable data. See here for more information about picklable data:
            https://docs.python.org/3/library/pickle.html#pickle-picklable.
        overwrite: Whether to overwrite the existing file if it exists.
    """
    mode = 'wb' if overwrite else 'xb'
    with open(filepath, mode) as pkl_file:
        pickle.dump(data, pkl_file, protocol=pickle.HIGHEST_PROTOCOL)


# ##################################################################################################
#                                   Geometry Helper Functions
# ##################################################################################################
def point_intersects_region(region: Sequence[int], point: Sequence[int]) -> bool:
    """Given a point, determines if it lies within a region

    Args:
        region: The region given as [x1, y1, x2, y2]
        point: The point to test (given as (x, y))

    Returns:
        True/False whether the point belongs to the region
    """
    return region[0] <= point[0] <= region[2] and region[1] <= point[1] <= region[3]


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


def convert_coordinates_to_mpp(coordinates, original_mpp, new_mpp, round_int=False):
    """Expresses box coordinates at a different resolution.

    Box coordinates are expected to be in [x1, y1, x2, y2] layout.

    Args:
        coordinates (list of 4 ints): The original box coordinates.
        original_mpp (float/2-tuple/list): The resolution corresponding to the original box
            coordinates (in microns per pixel). If a float, assumes mpp_x == mpp_y. If a tuple/list,
            give as (mpp_x, mpp_y).
        new_mpp (float/2-tuple/list): The new resolution (in microns per pixel). If a float,
            assumes mpp_x == mpp_y. If a tuple/list, give as (mpp_x, mpp_y).
        round_int (bool): Whether after conversion the result should be rounded and cast to int.

    Returns:
        coordinates (list of 4 ints): The coordinates in the new mpp space
    """
    # Ensure correct data format
    if not isinstance(original_mpp, (tuple, list)):
        original_mpp = (original_mpp, original_mpp)
    if not isinstance(new_mpp, (tuple, list)):
        new_mpp = (new_mpp, new_mpp)

    # Copy the coordinates
    coordinates = coordinates.copy()

    # Perform the conversion
    coordinates[0] = convert_pixel_mpp(coordinates[0], original_mpp[0], new_mpp[0], round_int=round_int)
    coordinates[1] = convert_pixel_mpp(coordinates[1], original_mpp[1], new_mpp[1], round_int=round_int)
    coordinates[2] = convert_pixel_mpp(coordinates[2], original_mpp[0], new_mpp[0], round_int=round_int)
    coordinates[3] = convert_pixel_mpp(coordinates[3], original_mpp[1], new_mpp[1], round_int=round_int)

    # Return the converted coordinates
    return coordinates


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
#                                       Drawing Functions
# ##################################################################################################
def draw_points_on_image(
    image: NDArray,
    points: Union[list, tuple, NDArray[float], NDArray[int]],
    colour: Tuple[int, int, int] = (0, 0, 0),
    radius: Optional[int] = 1,
    thickness: Optional[int] = -1,
    inplace: bool = False,
) -> NDArray:
    """Draws a point (circle)/multiple points (circles) on an image.

    Each point should be a 2-tuple with the x, y coordinates of the point centre.

    Args:
        image: Image to overlay rectangle on.
        points: The x, y circle centrepoint coordinate. For multiple points, pass in as a list.
        colour: RGB colour to overlay on image (default: black (0, 0, 0)).
        radius: Radius of the points to draw. If None, is derived by the smallest edge of the image.
        thickness: Thickness of the point border. If -1, the circle will be fully filled. If None,
            is derived by the smallest edge of the image.
        inplace: Whether the points should be drawn on the given image, or a copy made and a new
            image returned.
    """
    # Copy image
    if not inplace:
        image = image.copy()

    # Set up point representation (handling single/multiple points)
    if not isinstance(points[0], (tuple, list, np.ndarray)):
        points = [points]

    # Draw all points
    for idx, point in enumerate(points):
        point_colour = colour
        cv2.circle(
            image, (int(round(point[0])), int(round(point[1]))), radius=radius, color=point_colour,
            thickness=thickness, lineType=cv2.LINE_AA)

    return image
