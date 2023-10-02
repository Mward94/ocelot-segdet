"""Code from review_object_detection_metrics codebase: src.evaluators.coco_evaluator

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

    version ported from https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py
    Notes:
        1) The default area thresholds here follows the values defined in COCO, that is,
        small:           area <= 32**2
        medium: 32**2 <= area <= 96**2
        large:  96**2 <= area.
        If area is not specified, all areas are considered.
        2) COCO's ground truths contain an 'area' attribute that is associated with the segmented area if
        segmentation-level information exists. While coco uses this 'area' attribute to distinguish between
        'small', 'medium', and 'large' objects, this implementation simply uses the associated bounding box
        area to filter the ground truths.
        3) COCO uses floating point bounding boxes, thus, the calculation of the box area
        for IoU purposes is the simple open-ended delta (x2 - x1) * (y2 - y1).
        PASCALVOC uses integer-based bounding boxes, and the area includes the outer edge,
        that is, (x2 - x1 + 1) * (y2 - y1 + 1). This implementation assumes the open-ended (former)
        convention for area calculation.
"""
from collections import defaultdict
from typing import Optional

import numpy as np

from networks.metrics.rafaelpadilla.enumerators import BBFormat


def get_coco_metrics_points(
    groundtruth_bbs,
    detected_bbs,
    distance_threshold,
    area_range=(0, np.inf),
    max_dets: Optional[int] = 100,
):
    """ Calculate the Average Precision and Recall metrics based on COCO's official implementation.

    Here, a detection is deemed a True Positive if the Euclidean distance between itself and a
    ground truth box is <= the distance_threshold.

    Parameters
        ----------
            groundtruth_bbs : list
                A list containing objects of type BoundingBox representing the ground-truth bounding boxes.
            detected_bbs : list
                A list containing objects of type BoundingBox representing the detected bounding boxes.
            distance_threshold : float
                Euclidean distance threshold value used to consider a TP detection.
            area_range : (numerical x numerical)
                Lower and upper bounds on annotation areas that should be considered.
            max_dets :
                Upper bound on the number of detections to be considered for each class in an image.
                If None, uses all detections.
    Returns:
            A list of dictionaries. One dictionary for each class.
            The keys of each dictionary are:
            dict['class']: class representing the current dictionary;
            dict['precision']: array with the precision values;
            dict['recall']: array with the recall values;
            dict['F1']: array with the F1 values;
            dict['AP']: average precision;
            dict['interpolated precision']: interpolated precision values;
            dict['interpolated recall']: interpolated recall values;
            dict['total positives']: total number of ground truth positives;
            dict['TP']: total number of True Positive detections;
            dict['FP']: total number of False Positive detections;
            dict['scores']: sorted set of confidence scores;
            dict['cum tp']: cumulative count of true positives;
            dict['cum fp']: cumulative count of false positives;
            dict['best F1']: best F1 score achieved;
            dict['best F1 precision']: precision corresponding to best F1 score;
            dict['best F1 recall']: recall corresponding to best F1 score;
            dict['best F1 confidence']: confidence score threshold (inclusive) corresponding to best F1 score;
            if there was no valid ground truth for a specific class (total positives == 0),
            all the associated keys default to None
    """

    # separate bbs per image X class
    _bbs = _group_detections(detected_bbs, groundtruth_bbs)

    # pairwise euclidean distance
    _eucs = {k: _compute_eucs(**v) for k, v in _bbs.items()}

    # accumulate evaluations on a per-class basis
    _evals = defaultdict(lambda: {"scores": [], "matched": [], "NP": []})

    for img_id, class_id in _bbs:
        ev = _evaluate_image_points(
            _bbs[img_id, class_id]["dt"],
            _bbs[img_id, class_id]["gt"],
            _eucs[img_id, class_id],
            distance_threshold,
            max_dets,
            area_range,
        )
        acc = _evals[class_id]
        acc["scores"].append(ev["scores"])
        acc["matched"].append(ev["matched"])
        acc["NP"].append(ev["NP"])

    # now reduce accumulations
    for class_id in _evals:
        acc = _evals[class_id]
        acc["scores"] = np.concatenate(acc["scores"])
        acc["matched"] = np.concatenate(acc["matched"]).astype(bool)
        acc["NP"] = np.sum(acc["NP"])

    res = {}
    # run ap calculation per-class
    for class_id in _evals:
        ev = _evals[class_id]
        res[class_id] = {
            "class": class_id,
            **_compute_ap_recall(ev["scores"], ev["matched"], ev["NP"])
        }
    return res


def _group_detections(dt, gt):
    """ simply group gts and dts on a imageXclass basis """
    bb_info = defaultdict(lambda: {"dt": [], "gt": []})
    for d in dt:
        i_id = d.get_image_name()
        c_id = d.get_class_id()
        bb_info[i_id, c_id]["dt"].append(d)
    for g in gt:
        i_id = g.get_image_name()
        c_id = g.get_class_id()
        bb_info[i_id, c_id]["gt"].append(g)
    return bb_info


def _get_area(a):
    """ COCO does not consider the outer edge as included in the bbox """
    x, y, x2, y2 = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
    return (x2 - x) * (y2 - y)


def _compute_eucs(dt, gt):
    """ compute pairwise euclidean distances """

    eucs = np.zeros((len(dt), len(gt)))
    for g_idx, g in enumerate(gt):
        for d_idx, d in enumerate(dt):
            eucs[d_idx, g_idx] = _euclidean(d, g)
    return eucs


def _euclidean(a, b):
    """
    a and b are BoundingBox objects (defined in raphaelpadilla repo)

    To get their centre points, you'd need to first get their box coords, then find centre
    Then you could find Euclidean distance between the two
    """
    x1a, y1a, x2a, y2a = a.get_absolute_bounding_box(format=BBFormat.XYX2Y2)
    x1b, y1b, x2b, y2b = b.get_absolute_bounding_box(format=BBFormat.XYX2Y2)

    # Get centres for each box
    xac, yac = (x1a + x2a) / 2, (y1a + y2a) / 2
    xbc, ybc = (x1b + x2b) / 2, (y1b + y2b) / 2

    return np.linalg.norm(np.asarray([xac, yac]) - np.asarray([xbc, ybc]))


def _evaluate_image_points(dt, gt, eucs, distance_threshold, max_dets=None, area_range=None):
    """ use COCO's method to associate detections to ground truths

    This is modified to minimise euclidean distance instead of maximise IoU
    """
    # sort dts by increasing confidence
    dt_sort = np.argsort([-d.get_confidence() for d in dt], kind="stable")

    # sort list of dts and chop by max dets
    dt = [dt[idx] for idx in dt_sort[:max_dets]]
    eucs = eucs[dt_sort[:max_dets]]

    # generate ignored gt list by area_range
    def _is_ignore(bb):
        if area_range is None:
            return False
        return not (area_range[0] <= _get_area(bb) <= area_range[1])

    gt_ignore = [_is_ignore(g) for g in gt]

    # sort gts by ignore last
    gt_sort = np.argsort(gt_ignore, kind="stable")
    gt = [gt[idx] for idx in gt_sort]
    gt_ignore = [gt_ignore[idx] for idx in gt_sort]
    eucs = eucs[:, gt_sort]

    gtm = {}
    dtm = {}

    for d_idx, d in enumerate(dt):
        # information about best match so far (m=-1 -> unmatched)
        # Distance threshold value should be the largest of (0 or distance_threshold)
        # TP if distance < this value
        euc = max(distance_threshold, 1e-10)
        m = -1
        for g_idx, g in enumerate(gt):
            # if this gt already matched, and not a crowd, continue
            if g_idx in gtm:
                continue
            # if dt matched to reg gt, and on ignore gt, stop
            if m > -1 and gt_ignore[m] == False and gt_ignore[g_idx] == True:
                break
            # continue to next gt unless better match made
            if eucs[d_idx, g_idx] > euc:
                continue
            # if match successful and best so far, store appropriately
            euc = eucs[d_idx, g_idx]
            m = g_idx
        # if match made store id of match for both dt and gt
        if m == -1:
            continue
        dtm[d_idx] = m
        gtm[m] = d_idx

    # generate ignore list for dts
    dt_ignore = [
        gt_ignore[dtm[d_idx]] if d_idx in dtm else _is_ignore(d) for d_idx, d in enumerate(dt)
    ]

    # get score for non-ignored dts
    scores = [dt[d_idx].get_confidence() for d_idx in range(len(dt)) if not dt_ignore[d_idx]]
    matched = [d_idx in dtm for d_idx in range(len(dt)) if not dt_ignore[d_idx]]

    n_gts = len([g_idx for g_idx in range(len(gt)) if not gt_ignore[g_idx]])
    return {"scores": scores, "matched": matched, "NP": n_gts}


def _compute_ap_recall(scores, matched, NP, recall_thresholds=None):
    """ This curve tracing method has some quirks that do not appear when only unique confidence thresholds
    are used (i.e. Scikit-learn's implementation), however, in order to be consistent, the COCO's method is reproduced. """
    if NP == 0:
        return {
            "precision": None,
            "recall": None,
            "F1": None,
            "AP": None,
            "interpolated precision": None,
            "interpolated recall": None,
            "total positives": None,
            "TP": None,
            "FP": None,
            "scores": None,
            "cum tp": None,
            "cum fp": None,
            "best F1": None,
            "best F1 precision": None,
            "best F1 recall": None,
            "best F1 confidence": None,
        }

    # by default evaluate on 101 recall levels
    if recall_thresholds is None:
        recall_thresholds = np.linspace(0.0,
                                        1.00,
                                        int(np.round((1.00 - 0.0) / 0.01)) + 1,
                                        endpoint=True)

    # sort in descending score order
    inds = np.argsort(-scores, kind="stable")

    scores = scores[inds]
    matched = matched[inds]

    tp = np.cumsum(matched)
    fp = np.cumsum(~matched)

    rc = tp / NP
    pr = tp / (tp + fp)
    f1 = (2 * pr * rc) / (pr + rc + 1e-10)

    best_f1_idx = np.argmax(f1) if len(f1) > 0 else None

    # make precision monotonically decreasing
    i_pr = np.maximum.accumulate(pr[::-1])[::-1]

    rec_idx = np.searchsorted(rc, recall_thresholds, side="left")
    n_recalls = len(recall_thresholds)

    # get interpolated precision values at the evaluation thresholds
    i_pr = np.array([i_pr[r] if r < len(i_pr) else 0 for r in rec_idx])

    return {
        "precision": pr,
        "recall": rc,
        "F1": f1,
        "AP": np.mean(i_pr),
        "interpolated precision": i_pr,
        "interpolated recall": recall_thresholds,
        "total positives": NP,
        "TP": tp[-1] if len(tp) != 0 else 0,
        "FP": fp[-1] if len(fp) != 0 else 0,
        "scores": scores,
        "cum tp": tp,
        "cum fp": fp,
        "best F1": None if best_f1_idx is None else f1[best_f1_idx],
        "best F1 precision": None if best_f1_idx is None else pr[best_f1_idx],
        "best F1 recall": None if best_f1_idx is None else rc[best_f1_idx],
        "best F1 confidence": None if best_f1_idx is None else scores[best_f1_idx],
    }
