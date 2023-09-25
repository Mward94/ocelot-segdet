from typing import List, Dict, Any

import numpy as np
import torch
from scipy.spatial import KDTree
from torch import Tensor

from networks.postprocessors.postprocessor import Postprocessor
from util.constants import DET_POINTS_KEY, DET_INDICES_KEY, DET_SCORES_KEY


class PointNMS(Postprocessor):
    def __init__(self, euc_dist_threshold: float = 10, class_agnostic: bool = False):
        """Creates a postprocessor for non-maximum suppression (NMS) applied to points.

        This postprocessor is applicable to point detections.
        This is applied just like typical IoU-based NMS, except is based on Euclidean distance.

        As per typical NMS (And from the torchvision batched_nms docs):

        > Each index value correspond to a category, and NMS will not be applied between elements
          of different categories (unless class_agnostic specified).

        Args:
            euc_dist_threshold: Any points within a Euclidean distance of this threshold will be
                discarded.
            class_agnostic: If True, NMS will be applied across all classes.
        """
        super().__init__()
        self.euc_dist_threshold = euc_dist_threshold
        self.class_agnostic = class_agnostic

    @property
    def model_output_keys(self) -> List[str]:
        return [DET_POINTS_KEY, DET_INDICES_KEY, DET_SCORES_KEY]

    def postprocess(
        self,
        outputs: Dict[str, Any],
    ) -> Dict[str, Any]:
        new_outputs = outputs.copy()

        is_batched = isinstance(outputs[DET_POINTS_KEY], list)

        if is_batched:
            new_outputs[DET_POINTS_KEY] = []
            new_outputs[DET_INDICES_KEY] = []
            new_outputs[DET_SCORES_KEY] = []

            # We could potentially remove this loop by concatenating everything together and
            # tweaking the indices so that they are different for different examples in the batch.
            # It is unclear whether this would be faster or not.
            for det_points, det_indices, det_scores in zip(
                outputs[DET_POINTS_KEY],
                outputs[DET_INDICES_KEY],
                outputs[DET_SCORES_KEY]
            ):
                keep_indices = batched_point_nms(
                    points=det_points,
                    idxs=det_indices,
                    scores=det_scores,
                    euc_dist_threshold=self.euc_dist_threshold,
                    class_agnostic=self.class_agnostic,
                )

                new_outputs[DET_POINTS_KEY].append(det_points[keep_indices])
                new_outputs[DET_INDICES_KEY].append(det_indices[keep_indices])
                new_outputs[DET_SCORES_KEY].append(det_scores[keep_indices])
        else:
            keep_indices = batched_point_nms(
                points=outputs[DET_POINTS_KEY],
                idxs=outputs[DET_INDICES_KEY],
                scores=outputs[DET_SCORES_KEY],
                euc_dist_threshold=self.euc_dist_threshold,
                class_agnostic=self.class_agnostic,
            )

            new_outputs[DET_POINTS_KEY] = outputs[DET_POINTS_KEY][keep_indices]
            new_outputs[DET_INDICES_KEY] = outputs[DET_INDICES_KEY][keep_indices]
            new_outputs[DET_SCORES_KEY] = outputs[DET_SCORES_KEY][keep_indices]

        return new_outputs


def batched_point_nms(
        points: Tensor, scores: Tensor, idxs: Tensor, euc_dist_threshold: float,
        class_agnostic: bool = False,
) -> Tensor:
    """Performs point-based NMS based on Euclidean distance.

    Returns the indices of examples that should be retained.

    Each index value corresponds to a category. NMS will not be applied between elements of
    different categories.

    This implementation loosely follows the torchvision.ops.batched_nms implementation for boxes.

    Args:
        points: The [N, 2] set of points where NMS will be performed.
        scores: The [N,] set of scores for each of the points.
        idxs: The [N,] set of indices of the categories for each point.
        euc_dist_threshold: The Euclidean distance threshold. Points will be discarded that have
            Euclidean distance < euc_dist_threshold.
        class_agnostic: Whether NMS is applied between classes or not. Default = Not (individually
            in each class).

    Returns:
        The int64 tensor with indices of elements that have been kept by NMS, sorted in decreasing
        order of scores.
    """
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    if not class_agnostic:
        for class_id in torch.unique(idxs):
            curr_indices = torch.where(idxs == class_id)[0]
            curr_keep_indices = point_nms(points[curr_indices], scores[curr_indices], euc_dist_threshold)
            keep_mask[curr_indices[curr_keep_indices]] = True
    else:
        curr_keep_indices = point_nms(points, scores, euc_dist_threshold)
        keep_mask[curr_keep_indices] = True
    keep_indices = torch.where(keep_mask)[0]
    return keep_indices[scores[keep_indices].sort(descending=True)[1]]


def point_nms(points: Tensor, scores: Tensor, euc_dist_threshold: float) -> Tensor:
    """Performs point-based NMS on the points according to their Euclidean distances.

    NMS iteratively removes lower scoring points that have a Euclidean distance less than
    euc_dist_threshold with another (higher scoring) point.

    This implementation inserts all points in a KDTree and uses the KDTree to efficiently find
    nearby points to a given point.

    Args:
        points: The [N, 2] set of points where NMS will be performed.
        scores: The [N,] set of scores for each of the points.
        euc_dist_threshold: The Euclidean distance threshold. Points will be discarded that have
            Euclidean distance < euc_dist_threshold.

    Returns:
        The int64 tensor with indices of elements that have been kept by NMS, sorted in decreasing
        order of scores.
    """
    # Convert the scores to a numpy array
    scores = scores.numpy()

    # Sort the score indices in descending order.
    # Negating the scores is a trick to get descending order.
    # We could reverse by slicing ([::-1]), however this results in an array with negative stride
    # which can't be used for other operations without copying it first.
    descending_score_idxs = (-scores).argsort()

    # Create a mask of points that should be ignored
    ignore_scores_mask = np.zeros_like(scores, dtype=bool)

    # Store for the indices to be kept
    keep_indices = []

    # Sort the set of points by descending_score_idxs
    points = points[descending_score_idxs].numpy()

    # Construct a KDTree containing all points
    kd_tree = KDTree(points)

    # Iterate until we've handled every point
    while not np.all(ignore_scores_mask):
        # Find the index of the next point to add (based on score)
        next_idx = np.nonzero(~ignore_scores_mask)[0][0]

        # Add that index to the set of keep indices
        keep_indices.append(descending_score_idxs[next_idx])

        # Update the mask to remove the current point from analysis
        ignore_scores_mask[next_idx] = True

        # Find points close to the next point in the tree
        idxs = kd_tree.query_ball_point(points[next_idx], r=euc_dist_threshold)

        # Set points to ignore if they fall below the Euclidean distance threshold
        ignore_scores_mask[idxs] = True

    return torch.as_tensor(keep_indices, dtype=torch.int64)
