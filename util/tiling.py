"""Collection of functions related to tiling image data.
"""
import numpy as np


def generate_tiles(coordinates, tile_size, output_size):
    """Generates tile coordinates.

    Given absolute input coordinates for the region of interest this function generates tile
    coordinates relative to (x_min = 0 and y_min = 0) strided by output_size but with dimensions of
    tile_size.

    Three things to note:

    1. The entire region of interest is exactly covered in tiles in terms of output size. In the case the
      end tiles of each dimension does not fit the input region, a tile will be inserted at the
      end which occupies the space (end-size, end) coords along that dimension.

    2. In the case the output size is smaller than the tile size (input region size) it is assumed that
      padding will be added in order for the output region to fully cover in the region of interest.
      Negative coordinates are added to the left and bottom if paddings is need on the left and bottom
      respectively.

    3. If the region of interest coordinates is smaller than the tile size then it is assume padding
      will be added.


    Args:
        coordinates ([int]): A list of coords [x_min, y_min, x_max, y_max] to specify the area of
            interest.
        tile_size (tuple(int]): A tuple specifying the stride length for the tiling along the
            [height, width] dimensions respectively.
        output_size (tuple(int]): A tuple specifying the output length for the striding along the
            [height, width] dimensions respectively.

    Returns:
        all_tiles (list(np.array[int])): a list of numpy arrays one for each tile specified by
            [x_min, y_min, x_max, y_max]
    """
    if isinstance(tile_size, int):
        tile_size = (tile_size, tile_size)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)

    height_output_offsets, width_output_offsets = compute_tile_offsets_2D(coordinates, output_size)

    # If the output size is larger than the width/height
    # Force the tile coordinate to centre the tile about the coordinates
    #   In other words, generate negative padding
    if len(height_output_offsets) == 1 and height_output_offsets[0] == 0 and (
            coordinates[3] - coordinates[1]) < output_size[1]:
        height_output_offsets[0] = -(output_size[1] - (coordinates[3] - coordinates[1])) // 2
    if len(width_output_offsets) == 1 and width_output_offsets[0] == 0 and (
            coordinates[2] - coordinates[0]) < output_size[0]:
        width_output_offsets[0] = -(output_size[0] - (coordinates[2] - coordinates[0])) // 2

    all_tiles = []
    for height_idx in range(len(height_output_offsets)):
        for width_idx in range(len(width_output_offsets)):
            curr_height = height_output_offsets[height_idx]
            curr_width = width_output_offsets[width_idx]
            input_height, _ = inverse_center_crop_func_1D(curr_height, curr_height + output_size[0],
                                                          tile_size[0])
            input_width, _ = inverse_center_crop_func_1D(curr_width, curr_width + output_size[1],
                                                         tile_size[1])
            all_tiles.append(
                np.asarray([input_width, input_height, input_width + tile_size[1],
                            input_height + tile_size[0]]))

    return all_tiles


def compute_tile_offsets_2D(coordinates, stride_size):
    """Given an input region specified by coordinates this function computes the tile offsets in height
       and width dimensions with stride of stride_size.  Note in the case the end tiles of each dimension
       does not fit the input region, a tile will be inserted that at the end which occupies the space
       (end-size, end) coords along that dimension.

    Args:
        coordinates ([int]): A list of coords [x_min, y_min, x_max, y_max] that specifies the region
                             we want to perform the tiling over
        stride_size (tuple(int)) : A tuple specifying the stride length for the tiling along
                                   the [height, width] dimensions

    Returns:
        height_output_offsets: list of offsets along the height dimension
        width_output_offsets: list of offsets along the width dimension

    Examples:
        >>> compute_tile_offsets_2D([10, 5, 20, 25], (3, 2))
        ([0, 3, 6, 9, 12, 15, 17], [0, 2, 4, 6, 8])
    """

    # add the padding

    region_width = coordinates[2] - coordinates[0]
    region_height = coordinates[3] - coordinates[1]

    if (region_height > stride_size[0]):
        height_output_offsets = compute_tile_offsets_1D(region_height, stride_size[0])
    else:
        height_output_offsets = [0]
    if (region_width > stride_size[1]):
        width_output_offsets = compute_tile_offsets_1D(region_width, stride_size[1])
    else:
        width_output_offsets = [0]

    return height_output_offsets, width_output_offsets


def compute_tile_offsets_1D(length, stride_length):
    """Given the length along 1 dimension computes the offsets along that dimension with stride
    stride_length.

    Note in the case the end tile does not fit within length, a tile will be inserted at the end which
    occupies the space (end-size, end).

    Args:
        length (int): Specifies the length of the 1D region to do the tiling across.
        stride_length (int) : Specifies the stride length for the tiling.

    Returns:
        offsets: list of tile offsets computed

    Examples:
        >>> compute_tile_offsets_1D(16, 3)
        [0, 3, 6, 9, 12, 13]
    """

    offsets = list(range(0, length - stride_length, stride_length))
    if (offsets[-1] + stride_length < length):
        offsets.append(length - stride_length)

    return offsets


def inverse_center_crop_func_1D(output_start, output_end, input_length):
    """Given the start and end coords of an output 1D region (output_start and output_end)
       and the required input size, this function computes the coords of the input region that
       would result in the desired 1D output region.

    Args:
        output_start (int): Specifies the min coords for the desire output region.
        output_end (int) :Specifies the max coords for the desire output region.
        input_length (int): desired input length

    Returns:
        (input_start, input_end) (int): The min and max coords for the computed input region.

    Examples:
        >>> inverse_center_crop_func_1D(10, 20, 6)
        (12, 18)
    """
    gap_width = input_length - (output_end - output_start)
    input_start = output_start - int(gap_width / 2)
    input_end = input_start + input_length
    return (input_start, input_end)
