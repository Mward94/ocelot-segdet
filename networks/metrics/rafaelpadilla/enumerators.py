"""Code from review_object_detection_metrics codebase: src.utils.enumerators

Copied to this repo on 02/08/2021

@Article{electronics10030279,
    AUTHOR = {Padilla, Rafael and Passos, Wesley L. and Dias, Thadeu L. B. and Netto, Sergio L. and da Silva, Eduardo A. B.},
    TITLE = {A Comparative Analysis of Object Detection Metrics with a Companion Open-Source Toolkit},
    JOURNAL = {Electronics},
    VOLUME = {10},
    YEAR = {2021},
    NUMBER = {3},
    ARTICLE-NUMBER = {279},
    URL = {https://www.mdpi.com/2079-9292/10/3/279},
    ISSN = {2079-9292},
    DOI = {10.3390/electronics10030279}
}
"""
from enum import Enum


class MethodAveragePrecision(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.
        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    EVERY_POINT_INTERPOLATION = 1
    ELEVEN_POINT_INTERPOLATION = 2


class CoordinatesType(Enum):
    """
    Class representing if the coordinates are relative to the
    image size or are absolute values.
        Developed by: Rafael Padilla
        Last modification: Apr 28 2018
    """
    RELATIVE = 1
    ABSOLUTE = 2


class BBType(Enum):
    """
    Class representing if the bounding box is groundtruth or not.
    """
    GROUND_TRUTH = 1
    DETECTED = 2


class BBFormat(Enum):
    """
    Class representing the format of a bounding box.
    """
    XYWH = 1
    XYX2Y2 = 2
    PASCAL_XML = 3
    YOLO = 4


class FileFormat(Enum):
    ABSOLUTE_TEXT = 1
    PASCAL = 2
    LABEL_ME = 3
    COCO = 4
    CVAT = 5
    YOLO = 6
    OPENIMAGE = 7
    IMAGENET = 8
    UNKNOWN = 9
