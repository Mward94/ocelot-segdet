"""Utility functions for managing 3x3 matrices for cv2.warpAffine in pure numpy"""
import numpy as np


def identity():
    return np.eye(3, dtype=np.float64)


def affine(A=None, t=None):
    aff = identity()
    if A is not None:
        aff[0:2, 0:2] = A
    if t is not None:
        aff[0:2, 2] = t
    return aff


def rotate(theta):
    """Rotate counter-clockwise."""
    return affine(A=[[ np.cos(theta), np.sin(theta)],
                     [-np.sin(theta), np.cos(theta)]])


def rotate_around(rx, ry, theta):
    """
    Rotate counter-clockwise around (rx, ry)
    Rotating around (0, 0) in opencv is actually rotating around the centre
    of the first pixel.
    """
    return concatenate([
        translate(-rx, -ry),
        rotate(theta),
        translate(rx, ry)
    ])


def scale(sx, sy=None):
    """Scale."""
    if sy is None: sy = sx
    return affine(A=[[sx,  0],
                     [ 0, sy]])


def scale_around(srx, sry, sx, sy=None):
    """Scale around a different registration point"""
    return concatenate([
        translate(-srx, -sry),
        scale(sx, sy),
        translate(srx, sry),
    ])


def translate(tx, ty):
    """Translate."""
    return affine(t=[tx, ty])


def concatenate(matrices):
    matrix = identity()
    for m in reversed(matrices):
        matrix = matrix @ m
    return matrix


def homogeneous(coords):
    """
    Takes 2D coords  [N, ..., {x, y}] to homogeneous coords [N, ..., {x, y, 1}]
    """
    ones = np.ones((*coords.shape[:-1], 1))
    return np.concatenate([coords, ones], axis=-1)


def unhomogeneous(coords):
    """Inverse of homogeneous()"""
    # Normalise
    coords_norm = coords/coords[..., 2:]
    return coords_norm[..., :2]


def transform(mat, coords):
    """Batch transform `coords` shapend [..., 2]."""
    coords_np = homogeneous(np.asarray(coords))
    return unhomogeneous(coords_np @ mat.T)
