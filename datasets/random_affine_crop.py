"""Utilities for performing a randon affine transformation and crop.

There is one main class in this file: :class:`.RandomAffineCrop`. This can do most of what
Albumentations.Affine does but with a major distinction---it is able to calculate the region of the
input image required for the transform, which can be used to enable more efficient data loading.
The required input crop region can be calculated using :meth:`~.RandomAffineCrop.get_crop_bounds`.
"""
import decimal
import math
from typing import Union, Tuple, Sequence, Optional

import albumentations as A
import cv2
import numpy as np

from util import mat3
from util.helpers import get_region_dimensions


def safe_int(v, atol=7):
    """
    Simply casting to int truncates 0.999999999999 to 0.
    This accounts for floating point imprecisions.
    """
    return int(round(decimal.Decimal(v), atol))


def _translate_norm_matrix(img_size, crop_size, translate):
    iw, ih = img_size
    cw, ch = crop_size
    dx, dy = translate
    # Normalised translation is bounded by viable crop locations
    return mat3.translate((iw - cw + 1) * dx, (ih - ch + 1) * dy)


def get_pixel_transform(
        crop_size,
        translate=(0, 0),
        scale=1,
        rotate=0,
        scale_centre=(0.5, 0.5),
        rotate_centre=(0.5, 0.5),
        img_size=None,
):
    """Creates a 3x3 transformation matrix.

    For example, Given a bbox at (10, 10), rx=ry=angle=0, sc=1, dx=20, dy=40,
    applying this transform will move it to (30, 50).
    """
    cw, ch = crop_size
    rx, ry = rotate_centre
    srx, sry = scale_centre
    if img_size is None:
        t = mat3.translate(*translate)
    else:
        t = _translate_norm_matrix(img_size, crop_size, translate)
    # (rx, ry) and (srx, sry) are relative to the crop origin.
    s = mat3.scale_around(srx * cw, sry * ch, scale)
    r = mat3.rotate_around(rx * cw, ry * ch, np.deg2rad(rotate))
    M = mat3.concatenate([t, s, r])

    return M


def get_crop_pixel_transform(M, crop_size):
    """Converts a transformation matrix from whole image space into crop space.
    """
    xlo, ylo, _, _ = get_crop_bounds(M, crop_size)
    U = mat3.translate(xlo, ylo)
    Q = mat3.concatenate([U, M])
    return Q


def get_crop_bounds(M, crop_size):
    """Returns minimum pixel boundaries required to obtain pixel data.
    """
    # Assume crop is in top-right, then use M to move corners of crop
    # to correct place in pixel-space
    initial_corners = np.array([(0., 0), (1, 0), (1, 1), (0, 1)]) * crop_size
    corners = mat3.transform(np.linalg.inv(M), initial_corners)

    xlo, ylo = corners.min(axis=0)
    xhi, yhi = corners.max(axis=0)
    xlo, ylo = math.floor(xlo), math.floor(ylo)
    xhi, yhi = math.ceil(xhi), math.ceil(yhi)

    return xlo, ylo, xhi, yhi


def normalise_window(window):
    if len(window) == 4:
        xlo, ylo, xhi, yhi = window
        return (ylo, yhi), (xlo, xhi)
    return window


def get_slices(img_shp, window):
    """
    Slices for filling a crop from an image when cropping with `window`
    while accounting for the window being off the edge of `img`.
    *Note:* negative values in `window` are interpreted as-is, not as "from the end".
    """
    img_shp, window = np.array(img_shp), np.array(window)
    start = window[:, 0]
    end = window[:, 1]
    window_shp = end - start
    # Calculate crop slice positions
    crop_low = np.clip(0 - start, a_min=0, a_max=window_shp)
    crop_high = window_shp - np.clip(end - img_shp, a_min=0, a_max=window_shp)
    crop_slices = tuple(slice(low, high) for low, high in zip(crop_low, crop_high))
    # Calculate img slice positions
    start = np.clip(start, a_min=0, a_max=img_shp)
    end = np.clip(end, a_min=0, a_max=img_shp)
    img_slices = tuple(slice(low, high) for low, high in zip(start, end))
    return img_slices, crop_slices


def crop_read(img: np.ndarray, window, ch_axis=0) -> np.ndarray:
    """Gets from `img` a crop defined by `pos`/`size`.

    Note: `pos`/`size` may extend beyond image boundaries, and
    negative indices are considered as-is, not "from the end".

    Args:
        img: Image to pull raster data from.
        window: If in ((int, int), (int, int)) form, slices to pull from (y_window, x_window).
            If in (int, int, int, int) form, bounds (xlo, ylo, xhi, yhi).
        ch_axis: Which axis to place the channels in output.

    Returns:
        Crop raster data.
    """
    window = normalise_window(window)
    (ylo, yhi), (xlo, xhi) = window
    ysize, xsize = (yhi - ylo), (xhi - xlo)

    dtype = img.dtypes[0] if hasattr(img, 'dtypes') else img.dtype

    # Window may extend beyond the image boundaries, thus, instead of simply
    # stacking the outputs, which wouldn't be the full size, we create
    # a np.zeros and then paste into that the usable parts of the crop
    crop_shp = [ysize, xsize]
    img_shp = list(img.shape)
    if len(img.shape) == 3:
        # Note: pop drops the channel from img_shp
        crop_shp.insert(ch_axis, img_shp.pop(ch_axis))
    crop = np.zeros(crop_shp, dtype=dtype)

    img_slices, crop_slices = get_slices(img_shp, window)

    # Slices are for just [H, W]
    # If there's a channel dimension, insert a None for it in the slices
    if len(img.shape) == 3:
        crop_slices = list(crop_slices)
        img_slices = list(img_slices)
        crop_slices.insert(ch_axis, slice(None))
        img_slices.insert(ch_axis, slice(None))
    crop[tuple(crop_slices)] = img[tuple(img_slices)]

    return crop


def fetch(img, M, crop_size, interpolate=cv2.INTER_LINEAR, skip_image_crop=False):
    # Find M, relative to the crop
    Q = get_crop_pixel_transform(M, crop_size)

    if skip_image_crop:
        crop = img
    else:
        # Get crop and return selected data
        bounds = get_crop_bounds(M, crop_size)
        crop = crop_read(img, bounds, ch_axis=2)

    # cv2.warpAffine treats (0,0) to be the centre of the top-left pixel. This is not a bug; it is
    # necessary for proper sampling. However, this can cause unexpected mis-alignments between
    # bboxes and the sampled image.
    half_in = mat3.translate(1 / 2, 1 / 2)
    half_out = mat3.translate(-1 / 2, -1 / 2)
    P = mat3.concatenate([half_in, Q, half_out])

    return cv2.warpAffine(crop, P[:2], crop_size,
                          flags=interpolate, borderMode=cv2.BORDER_CONSTANT)


class RandomAffineCrop(A.DualTransform):
    name: str = 'RandomAffineCrop'
    #: Size of crop in pixels as (width, height).
    crop_size: Tuple[int, int]

    def __init__(
            self,
            crop_size: Union[int, Sequence[int]],
            translate: Union[float, Sequence[float]] = (-1, 0),
            scale: Union[float, Sequence[float]] = (0.9, 1.1),
            rotate: Union[float, Sequence[float]] = (-180, 180),
            scale_centre: Sequence[float] = (0.5, 0.5),
            rotate_centre: Sequence[float] = (0.5, 0.5),
            scale_p: float = 1.0,
            rotate_p: float = 1.0,
            bbox_rotate_method: str = 'ellipse',
            always_apply: bool = False,
            p: float = 1.0,
    ):
        """Applies a random affine transformation.

        Logically, operations are performed in the following order:

        1. translation
        2. scaling
        3. rotation
        4. cropping

        The transformation parameters are represented as a params dictionary::

            {
                'translate': (dx, dy),
                'scale': scale,
                'rotate': angle,
                'scale_centre': (srx, sry),
                'rotate_centre': (rx, ry),
            }

        * dx, dy are normalised to IMAGE dimensions
        * srx, sry are normalised to CROP dimensions
        * rx, ry are normalised to CROP dimensions
        * angle is in degrees

        You can generate a dictionary like this with random values by using :meth:`.get_params` or
        :meth:`.get_params_safe`.

        You may provide the random distribution for each operation as a callable
        that takes no parameters.

        Args:
            crop_size: The (width, height) of the output crop.
            translate: Fixed translation value or range of translation values to randomly sample
                from. Units: normalised IMAGE dimensions between -1 (right/bottom) and 0 (left/top).
            scale: Fixed scale factor or range of scale factors to randomly sample from in log
                space. Units: percentage, where 1.0 means no scaling.
            rotate: Fixed rotation angle or range of rotation angles to randomly sample from.
                Units: degrees, positive is counter-clockwise.
            scale_p: The probability of scaling the image.
            rotate_p: The probability of rotating the image.
            bbox_rotate_method: rotation method used for the bounding boxes. Should be one of
                "largest_box" or "ellipse".
            always_apply:
            p: Probability of applying the transform. You should leave this at the default value
                of 1.0 unless you have a very good reason to change it.
        """
        super().__init__(always_apply, p)
        if isinstance(crop_size, int):
            crop_size = (crop_size, crop_size)
        else:
            crop_size = tuple(crop_size)
        self.crop_size = crop_size

        self._translate = translate
        self._scale = scale
        self._rotate = rotate
        self.scale_p = scale_p
        self.rotate_p = rotate_p
        self.scale_centre = scale_centre
        self.rotate_centre = rotate_centre
        if bbox_rotate_method not in {'largest_box', 'ellipse'}:
            raise ValueError(f'Bounding box rotation method {self.method} is not valid.')
        self.bbox_rotate_method = bbox_rotate_method

        self.disable_smart_loading()

    def sample_translate(self) -> Tuple[float, float]:
        """Samples the 'translate' transformation parameter.
        """
        if isinstance(self._translate, (int, float)):
            return (self._translate, self._translate)
        return tuple(np.random.uniform(self._translate[0], self._translate[1], (2,)))

    def sample_scale(self) -> float:
        """Samples the 'scale' transformation parameter.
        """
        if np.random.uniform() >= self.scale_p:
            return 1.0
        if isinstance(self._scale, (int, float)):
            return self._scale
        # If you sample in the range (0.5, 2) then you are more likely to sample a value greater than 1
        # than less than 1 (i.e. enlarging the image is more likely than shrinking). By sampling in log
        # space we avoid this problem.
        return np.exp(np.random.uniform(np.log(self._scale[0]), np.log(self._scale[1])))

    def sample_rotate(self) -> float:
        """Samples the 'rotate' transformation parameter.
        """
        if np.random.uniform() >= self.rotate_p:
            return 0.0
        if isinstance(self._rotate, (int, float)):
            return self._rotate
        return np.random.uniform(self._rotate[0], self._rotate[1])

    def is_smart_loading_enabled(self) -> bool:
        return self._frozen_M is not None

    def enable_smart_loading(
            self,
            region_width: int,
            region_height: int,
            params: Optional[dict] = None,
    ):
        """Samples transform parameters, freezes them, and enables "smart loading".

        Once smart loading is enabled, instead of accepting the full region_width x region_height
        image as input to the transform, the transformation matrix will be adjusted such that it
        expects a cropped window instead. It is up to the user of this transform to use the crop
        window returned by this method to crop the input. This applies to bounding box and keypoint
        annotations also.

        Args:
            region_width: The full width of the region.
            region_height: The full height of the region.
            params: The transformation parameters. If unspecified, the parameters will be
                obtained using :meth:`.get_params`.

        Returns:
            The crop window and dictionary of attributes to set to disable smart loading.
        """
        # Ensure _frozen_M not already defined (invalid use of this function)
        if self.is_smart_loading_enabled():
            raise RuntimeError('Smart loading cannot be enabled twice without resetting the '
                               'transform parameters.')

        # Get params of transform then obtain M (this will become fixed)
        if params is None:
            params = self.get_params()
        M = self.get_pixel_transform((region_width, region_height), **params)

        # Given M, determine the window to load
        window = self.get_crop_bounds(M)

        # Fix the M matrix
        self_transform_reset_params = {'_frozen_M': None, '_frozen_window': None}
        self._frozen_M = M
        self._frozen_window = window

        return window, self_transform_reset_params

    def disable_smart_loading(self):
        self._frozen_M = None
        self._frozen_window = None

    def get_params(self):
        """Gets a set of random affine parameters.
        """
        return {
            'translate': self.sample_translate(),
            'scale': self.sample_scale(),
            'rotate': self.sample_rotate(),
            'scale_centre': self.scale_centre,
            'rotate_centre': self.rotate_centre,
        }

    def get_params_safe(self, image_size):
        """
        Gets a set of params sampled uniformly from provided ranges, while
        ensuring crop is within bounds of the image
        """
        # TODO: How can you make this safe, efficiently?
        #       Central square can be sampled normally.
        #       Triangles on the sides: trigonometry to determine how much are
        #           valid crop locations
        #       Select between by area
        # In practice, for large images, this is very unlikely to occur more than once
        for i in range(100):
            params = self.get_params()
            M = get_pixel_transform(self.crop_size, img_size=image_size, **params)
            if self.check_sample_inside(M, image_size):
                return params

        raise Exception('Could\'t spatial params for crop within image. Check image and crop sizes')

    def check_sample_inside(self, M, image_size):
        """ Returns true if this will spatial wholly within the image """
        iw, ih = image_size
        xlo, ylo, xhi, yhi = self.get_crop_bounds(M)
        return xlo >= 0 and ylo >= 0 and xhi < iw and yhi < ih

    def get_pixel_transform(self, img_size, translate=(0, 0), scale=1, rotate=0,
                            scale_centre=(0.5, 0.5), rotate_centre=(0.5, 0.5)):
        if self.is_smart_loading_enabled():
            window_size = get_region_dimensions(self._frozen_window)
            if img_size != window_size:
                raise RuntimeError(f'Smart loading has been enabled and dictates that the '
                                   f'transform input has size {window_size}. Got {img_size}.')
            # NOTE: params like translate, scale, etc are ignored when smart loading is enabled
            # (M is frozen).
            return self._frozen_M
        return get_pixel_transform(self.crop_size, translate, scale, rotate, scale_centre,
                                   rotate_centre, img_size)

    def get_crop_pixel_transform(self, M):
        return get_crop_pixel_transform(M, self.crop_size)

    def get_crop_bounds(self, M):
        return get_crop_bounds(M, self.crop_size)

    def fetch(self, img, interpolate, M=None, **params):
        img_size = img.shape[1::-1]

        if M is None:
            M = self.get_pixel_transform(img_size, **params)

        return fetch(img, M, self.crop_size, interpolate,
                     skip_image_crop=self.is_smart_loading_enabled())

    def apply(self, img, rows=None, cols=None, **params):
        return self.fetch(img, interpolate=cv2.INTER_LINEAR, **params)

    def apply_to_mask(self, img, rows=None, cols=None, **params):
        return self.fetch(img, interpolate=cv2.INTER_NEAREST, **params)

    def apply_to_bbox(self, bbox, rows, cols, M=None, **params):
        """
        Applies the transform to the bbox.

        Note: takes axis-aligned coordinates, returns axis-aligned coordinates. Thus, if there's a
        rotation,

        If self.bbox_rotate_method is largest_box:

            The final axis-aligned corners are the minimum axis-aligned box to cover the rotated
            corners and the area gets larger.
            i.e. rotating 45 degrees and then rotating -45 degrees won't give
            you the same bbox.

        If self.bbox_rotate_method is ellipse:

            The final axis-aligned corners are those as described in the paper 'Towards Rotation
            Invariance in Object Detection': https://arxiv.org/pdf/2109.13488.pdf
        """
        if M is None:
            M = self.get_pixel_transform((cols, rows), **params)
        xlo, ylo, xhi, yhi = bbox

        if self.is_smart_loading_enabled():
            offx, offy, _, _ = self.get_crop_bounds(M)
            U = mat3.translate(offx, offy)
            M = mat3.concatenate([U, M])

        # Apply transform and re-align corners with axis
        if self.bbox_rotate_method == 'largest_box':
            # Translate to pixel values
            pxlo = xlo * cols
            pylo = ylo * rows
            pxhi = xhi * cols
            pyhi = yhi * rows
            initial_corners = [(pxlo, pylo), (pxhi, pylo), (pxlo, pyhi), (pxhi, pyhi)]
        elif self.bbox_rotate_method == 'ellipse':
            w, h = (xhi - xlo) / 2, (yhi - ylo) / 2
            data = np.arange(0, 360, dtype=np.float32)
            x = cols * (w * np.sin(np.radians(data)) + (w + xlo))
            y = rows * (h * np.cos(np.radians(data)) + (h + ylo))
            initial_corners = [(a, b) for a, b in zip(x, y)]
        else:
            raise ValueError(f'Invalid bbox rotation method: {self.method}')

        corners = mat3.transform(M, initial_corners)
        npxlo, npylo = corners.min(axis=0)
        npxhi, npyhi = corners.max(axis=0)

        # Normalise to crop size
        nxlo = npxlo / self.crop_size[0]
        nylo = npylo / self.crop_size[1]
        nxhi = npxhi / self.crop_size[0]
        nyhi = npyhi / self.crop_size[1]

        return nxlo, nylo, nxhi, nyhi

    def apply_to_keypoint(self, keypoint, rows, cols, M=None, **params):
        """
        Keypoints can actually be a vector; they start somewhere and point somewhere else.
            https://albumentations.ai/docs/getting_started/keypoints_augmentation/
        """
        if M is None:
            M = self.get_pixel_transform((cols, rows), **params)
        # Get point and vector
        x, y, angle, scale = keypoint
        vx = math.cos(angle) * scale
        vy = -math.sin(angle) * scale

        if self.is_smart_loading_enabled():
            offx, offy, _, _ = self.get_crop_bounds(M)
            U = mat3.translate(offx, offy)
            M = mat3.concatenate([U, M])

        # Transform point and vector in pixel space
        points = [[x, y], [x + vx, y + vy]]
        # New x,y denoted nx,ny.
        (nx, ny), (nevx, nevy) = mat3.transform(M, points)
        # Vector is relative to nx, ny
        nvx = nevx - nx
        nvy = nevy - ny
        nscale = math.sqrt(nvx ** 2 + nvy ** 2)
        nangle = math.atan2(nvy, nvx)
        # Albumentations follows the convention that a positive rotation
        # is counter-clockwise for the angle of the keypoint
        nangle = -nangle

        # Return vector in polar form
        # To match with `albumentations.Rotate`, we take the int part.
        return safe_int(nx), safe_int(ny), nangle, nscale
