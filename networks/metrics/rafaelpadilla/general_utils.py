"""Code from review_object_detection_metrics codebase: src.utils.general_utils

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

This code was adapted to only include the required functions
"""


# size => (width, height) of the image
# box => (X1, X2, Y1, Y2) of the bounding box
def convert_to_relative_values(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    cx = (box[1] + box[0]) / 2.0
    cy = (box[3] + box[2]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = cx * dw
    y = cy * dh
    w = w * dw
    h = h * dh
    # YOLO's format
    # x,y => (bounding_box_center)/width_of_the_image
    # w => bounding_box_width / width_of_the_image
    # h => bounding_box_height / height_of_the_image
    return (x, y, w, h)


# size => (width, height) of the image
# box => (centerX, centerY, w, h) of the bounding box relative to the image
def convert_to_absolute_values(size, box):
    w_box = size[0] * box[2]
    h_box = size[1] * box[3]

    x1 = (float(box[0]) * float(size[0])) - (w_box / 2)
    y1 = (float(box[1]) * float(size[1])) - (h_box / 2)
    x2 = x1 + w_box
    y2 = y1 + h_box
    return (round(x1), round(y1), round(x2), round(y2))
