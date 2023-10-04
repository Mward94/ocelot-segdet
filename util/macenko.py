"""Utilities for normalising a H&E image using the Macenko method.

The steps for Macenko colour normalisation are as follows:

1. Convert RGB to OD.
2. Remove data with OD intensity less than β.
3. Calculate singular value decomposition (SVD) on the OD tuples.
4. Create plane from the SVD directions corresponding to the two largest singular values.
5. Project data onto the plane, and normalize to unit length.
6. Calculate angle of each point wrt the first SVD direction.
7. Find robust extremes (αth and (100−α)th 7 percentiles) of the angle.
8. Convert extreme values back to OD space.

References:

* https://www.youtube.com/watch?v=yUrwEYgZUsA
* https://github.com/bnsreenu/python_for_microscopists/blob/master/122_normalizing_HnE_images.py
* https://www.kaggle.com/robotdreams/stain-normalization
* https://github.com/schaugf/HEnorm_python/blob/master/normalizeStaining.py
"""

from typing import Optional, Union, Dict, Any

import numpy as np
from numpy.typing import NDArray


#: Default reference H&E optical density matrix.
DEFAULT_HE_REF = np.array([[0.5626, 0.2159],
                           [0.7201, 0.8012],
                           [0.4062, 0.5581]], dtype=np.float32)

#: Default reference maximum stain concentrations for H&E.
DEFAULT_MAX_C_REF = np.array([1.9705, 1.0308], dtype=np.float32)


def _chunked_matmul(a: NDArray, b: NDArray, chunk_size: int):
    """Matrix product of two arrays, computed in chunks.
    """
    assert a.dtype == b.dtype
    n = a.shape[0]
    m = b.shape[1]
    out = np.empty((n, m), dtype=a.dtype)
    for i in range(0, n, chunk_size):
        for j in range(0, m, chunk_size):
            out[i:i + chunk_size, j:j + chunk_size] = np.matmul(
                a[i:i + chunk_size, :], b[:, j:j + chunk_size])
    return out


def rgb_to_od(rgb: NDArray[np.uint8], Io: int = 240) -> NDArray[np.float32]:
    """Converts RGB pixel data to optical density.

    .. math::
        \mathit{OD} = -\log_{10}(\dfrac{\mathit{RGB} + 1}{\mathit{Io}})

    Args:
        rgb: RGB pixel data.
        Io: Transmitted light intensity (normalizing factor for image intensities).

    Returns:

    """
    OD = np.array(rgb, dtype=np.float32)
    OD += 1  # Avoids log(0)
    OD /= Io
    np.log10(OD, out=OD)
    np.negative(OD, out=OD)
    return OD


def calculate_he_matrix(
    OD: NDArray[np.float32],
    alpha: float = 1,
    beta: float = 0.15,
    chunk_size: int = 65536,
) -> NDArray[np.float32]:
    """Calculate the HE matrix according to the Macenko normalisation method.

    Args:
        OD: Image pixel values in OD (optical density) space.
        alpha: Tolerance for the pseudo-min and pseudo-max when finding robust extremes of the
            angle.
        beta: OD threshold for transparent pixels.
        chunk_size: The maximum chunk size to use when performing certain calculations like least
            squares. This lowers peak memory requirements and avoids limitations of the backend
            math libraries.

    Returns:
        The computed HE matrix following SVD.
    """
    # Remove transparent pixels (clear region with no tissue) by removing data with OD intensity
    # less than beta.
    OD_hat = OD[~np.any(OD < beta, axis=1)]

    # Estimate covariance matrix of ODhat (transposed) then compute eigenvalues/vectors.
    eigvecs = np.linalg.eigh(np.cov(OD_hat.T))[1].astype(np.float32)

    # Project on the plane spanned by eigenvectors corresponding to two largest eigenvalues.
    T_hat = _chunked_matmul(OD_hat, eigvecs[:, 1:3], chunk_size)
    del OD_hat

    # Find min/max vectors and project back to OD space
    phi = np.arctan2(T_hat[:, 1], T_hat[:, 0])
    del T_hat
    min_phi, max_phi = np.percentile(phi, (alpha, 100 - alpha))
    v_min = np.matmul(eigvecs[:, 1:3],
                      np.asarray([[np.cos(min_phi)], [np.sin(min_phi)]], dtype=np.float32))
    v_max = np.matmul(eigvecs[:, 1:3],
                      np.asarray([[np.cos(max_phi)], [np.sin(max_phi)]], dtype=np.float32))

    # Heuristic to make vector corresponding to haematoxylin first and eosin second.
    if v_min[0] > v_max[0]:
        HE = np.stack((v_min[:, 0], v_max[:, 0]), axis=-1)
    else:
        HE = np.stack((v_max[:, 0], v_min[:, 0]), axis=-1)

    return HE


def calculate_concentrations(
    OD: NDArray[np.float32],
    HE: NDArray[np.float32],
    chunk_size: int = 65536,
) -> NDArray[np.float32]:
    """Calculates concentrations of the individual H&E stains.

    Args:
        OD: Image pixel values in OD (optical density) space.
        HE: The pre-computed HE matrix (see :func:`calculate_he_matrix`).
        chunk_size: The maximum chunk size to use when performing certain calculations like least
            squares. This lowers peak memory requirements and avoids limitations of the backend
            math libraries.

    Returns:
        Concentrations of individual stains, `C`.
    """
    # Rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # We compute the least squares in chunks to avoid size issues with the backend (and to lower
    # peak memory requirements).
    C = np.empty((HE.shape[1], Y.shape[1]), dtype=np.float32)
    for i in range(0, C.shape[1], chunk_size):
        C[:, i:i+chunk_size], _, rank_i, _ = np.linalg.lstsq(HE, Y[:, i:i+chunk_size], rcond=-1)
        # Ensure the rank is 2 (otherwise normalisation has failed)
        if rank_i != 2:
            raise RuntimeError(
                f'Least squares failed. Expected matrix of rank 2 from least squares. '
                f'Got matrix of rank: {rank_i}.')

    return C


def normalised_concentrations_to_rgb(
    C_norm: NDArray[np.float32],
    ret_img: str = 'HE',
    Io: int = 240,
    chunk_size: int = 65536,
    HE_ref: NDArray[np.float32] = DEFAULT_HE_REF,
) -> NDArray[np.uint8]:
    """Calculates pixel values in RGB space using a reference HE mixing matrix.

    Args:
        C_norm: Normalised stain concentrations.
        ret_img: The type of image to create (``"H"`` = haematoxylin only, ``"E"`` = eosin only,
            ``"HE"`` = haematoxylin and eosin).
        Io: Transmitted light intensity (normalizing factor for image intensities).
        chunk_size: The maximum chunk size to use when performing certain calculations like least
            squares. This lowers peak memory requirements and avoids limitations of the backend
            math libraries.
        HE_ref: The reference HE matrix for normalisation.

    Returns:
        Pixel values in RGB space.
    """
    if ret_img == 'H':
        stain_ref = np.expand_dims(-HE_ref[:, 0], axis=1)
        C_norm = np.expand_dims(C_norm[0, :], axis=0)
    elif ret_img == 'E':
        stain_ref = np.expand_dims(-HE_ref[:, 1], axis=1)
        C_norm = np.expand_dims(C_norm[1, :], axis=0)
    else:  # ret_img == 'HE'
        stain_ref = -HE_ref
    img = _chunked_matmul(stain_ref, C_norm, chunk_size)
    np.exp(img, out=img)
    img *= Io

    # Bound to <= 255
    np.clip(img, a_min=None, a_max=255, out=img)

    return img.astype(np.uint8)


def precompute_imagewide_normalisation_parameters(
    image: NDArray[np.uint8],
    Io: int = 240,
    alpha: float = 1,
    beta: float = 0.15,
    chunk_size: int = 65536,
) -> Dict[str, Any]:
    OD = rgb_to_od(image.reshape((-1, 3)), Io=Io)
    HE = calculate_he_matrix(OD, alpha=alpha, beta=beta, chunk_size=chunk_size)
    C = calculate_concentrations(OD, HE, chunk_size=chunk_size)
    max_C = np.percentile(C, 99, axis=1)
    return {
        'Io': Io,
        'alpha': alpha,
        'beta': beta,
        'HE': HE,
        'max_C': max_C,
    }


def normalise_he_image_(
    image: NDArray[np.uint8],
    Io: int = 240,
    alpha: float = 1,
    beta: float = 0.15,
    ret_img: str = 'HE',
    mask: Optional[NDArray[np.bool_]] = None,
    chunk_size: int = 65536,
    HE: Optional[NDArray[np.float32]] = None,
    max_C: Optional[NDArray[np.float32]] = None,
    HE_ref: NDArray[np.float32] = DEFAULT_HE_REF,
    max_C_ref: NDArray[np.float32] = DEFAULT_MAX_C_REF,
) -> Dict[str, Any]:
    """Colour normalizes a H&E image using the Macenko method.

    This function will modify the image in-place.

    Depending on the value of `ret_img`, this will return either the colour-normalized H&E image
    (``"HE"``), the deconvolved haematoxylin stain (``"H"``), or the deconvolved eosin stain
    (``"E"``).

    Args:
        image: Image to be colour normalized (expects an RGB image with HWC axis order).
        Io: Transmitted light intensity (normalizing factor for image intensities).
        alpha: Tolerance for the pseudo-min and pseudo-max (Step 7).
        beta: OD threshold for transparent pixels (Step 2).
        ret_img: The type of image to return (``"HE"``, ``"H"`` or ``"E"``).
        mask: The mask of pixels to be included in the normalisation.
        chunk_size: The maximum chunk size to use when performing certain calculations like least
            squares. This lowers peak memory requirements and avoids limitations of the backend
            math libraries.
        HE: The pre-computed HE matrix relating to the area that this image comes from. This can be
            used to pre-compute the HE matrix on a large area, then apply it to smaller areas
            on-the-fly. If specifying, must also specify max_C.
        max_C: The pre-computed max_C matrix relating to the area that this image comes from. This
            can be used to pre-compute the max_C matrix on a large area, then apply it to smaller
            areas on-the-fly. If specifying, must also specify HE.
        HE_ref: The reference HE matrix for normalisation.
        max_C_ref: The reference maximum concentrations for normalisation.

    Returns:
        Normalisation parameters including the computed HE matrix following SVD and the computed
        max_C matrix following least-squares. If these values were passed in as arguments they will
        be returned unchanged.

    Raises:
        np.linalg.LinAlgError: If the eigenvalue computation does not converge.
    """
    # Validate ret_img
    if ret_img not in ('HE', 'H', 'E'):
        raise ValueError(f'ret_img should be: \'HE\', \'H\' or \'E\'. Is: {ret_img}')

    # Validate HE/max_C
    if (HE is not None and max_C is None) or (HE is None and max_C is not None):
        raise ValueError('Must specify either BOTH HE/max_C or neither.')

    # Mask the image (if specified)
    if mask is not None:
        used_image = image[mask]
    else:
        used_image = image

    # Reshape into a vector with 3 columns
    used_image = used_image.reshape((-1, 3))

    # Step 1: Convert RGB to optical density
    OD = rgb_to_od(used_image, Io=Io)

    # Support pre-computed HE
    if HE is None:
        # Steps 2--7: Compute HE matrix
        HE = calculate_he_matrix(OD, alpha=alpha, beta=beta, chunk_size=chunk_size)

    # Determine concentrations of the individual stains
    C = calculate_concentrations(OD, HE, chunk_size=chunk_size)

    # Normalise stain concentrations
    if max_C is None:
        max_C = np.percentile(C, 99, axis=1).astype(np.float32)
    C_norm_factor = np.divide(max_C, max_C_ref)
    np.divide(C, C_norm_factor[:, np.newaxis], out=C)

    # Step 8: Convert extreme values back to OD space.
    img = normalised_concentrations_to_rgb(C, ret_img=ret_img, Io=Io, chunk_size=chunk_size,
                                           HE_ref=HE_ref)

    # Handle the mask
    if mask is None:
        image[:] = np.reshape(img.T, image.shape)
    else:
        image.reshape((-1, 3))[mask.ravel()] = img.T

    return {
        'Io': Io,
        'alpha': alpha,
        'beta': beta,
        'HE': HE,
        'max_C': max_C,
    }


def normalise_he_image(
    image: Union[NDArray[np.uint8]],
    Io: int = 240,
    alpha: float = 1,
    beta: float = 0.15,
    ret_img: str = 'HE',
    mask: Optional[NDArray[bool]] = None,
    chunk_size: int = 65536,
    HE: Optional[NDArray[np.float32]] = None,
    max_C: Optional[NDArray[np.float32]] = None,
    HE_ref: NDArray[np.float32] = DEFAULT_HE_REF,
    max_C_ref: NDArray[np.float32] = DEFAULT_MAX_C_REF,
) -> NDArray[np.uint8]:
    """Colour normalizes a H&E image using the Macenko method.

    See Also:
        * :func:`normalise_he_image_`

    Returns:
        The colour normalized H&E image or extracted H/E stain. This is returned as an RGB
        with HWC axis.
    """
    out_image = image.copy()
    normalise_he_image_(out_image, Io, alpha, beta, ret_img, mask, chunk_size, HE, max_C,
                        HE_ref, max_C_ref)
    return out_image
