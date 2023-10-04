"""Postprocessor that performs blob detection on an output point heatmap.
"""
from typing import Optional, List, Dict, Any, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

from networks.postprocessors.postprocessor import Postprocessor
from util.constants import POINT_HEATMAP_KEY, DET_POINTS_KEY, DET_INDICES_KEY, DET_SCORES_KEY

DEFAULT_BLOB_PARAMS = dict(
    filterByColor=True,
    blobColor=255,
    filterByArea=True,
    minArea=25,
    maxArea=1500,
    filterByCircularity=False,
    filterByConvexity=True,
    filterByInertia=True,
    minThreshold=50,
    maxThreshold=220,
    thresholdStep=10,
    minDistBetweenBlobs=16,
    minRepeatability=2,
)


class PointHeatmapBlobDetector(Postprocessor):
    def __init__(self, blob_params: Optional[Dict[str, Any]] = None,
                 ignore_index: Optional[int] = None):
        """Creates a postprocessor for turning a heatmap into point detections/classes/scores.

        Per-heatmap channel, the blob detector is applied to attempt to convert heatmap predictions
        into points. If a point is found, the class index is generated based on the heatmap channel
        the point was found in, and the score is found by looking at the value in the heatmap at
        the found point.

        A set of sensible default values for the blob detector are used by default, however all of
        these can be overridden by the blob_params parameter.

        Following postprocessing, the following keys will be added to the output dictionary:

        * PRED_DET_POINTS_KEY

        * PRED_DET_INDICES_KEY

        * PRED_DET_SCORES_KEY

        Args:
            blob_params: Parameters to use to configure the blob detector.
            ignore_index: The index of a channel in the heatmap to ignore.
        """
        super().__init__()
        self.ignore_index = ignore_index

        update_params = DEFAULT_BLOB_PARAMS.copy()
        if blob_params is not None:
            update_params.update(blob_params)

        # Set up blob detector
        params = cv2.SimpleBlobDetector_Params()

        for param_key, param_val in update_params.items():
            if not hasattr(params, param_key):
                raise AttributeError(f'Key \'{param_key}\' not supported by blob detector.')
            setattr(params, param_key, param_val)

        self.detector = cv2.SimpleBlobDetector_create(params)

    @property
    def model_output_keys(self) -> List[str]:
        return [POINT_HEATMAP_KEY]

    def postprocess(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        new_outputs = outputs.copy()

        # Batched if 4 dims (B, Nc, H, W), otherwise 3 (Nc, H, W)
        is_batched = outputs[POINT_HEATMAP_KEY].ndim == 4

        if is_batched:
            new_outputs[DET_POINTS_KEY] = []
            new_outputs[DET_INDICES_KEY] = []
            new_outputs[DET_SCORES_KEY] = []

            for batch_heatmap in outputs[POINT_HEATMAP_KEY]:
                points, indices, scores = self.detect_points_in_heatmap(batch_heatmap)

                # Append to the outputs
                new_outputs[DET_POINTS_KEY].append(points)
                new_outputs[DET_INDICES_KEY].append(indices)
                new_outputs[DET_SCORES_KEY].append(scores)
        else:
            new_outputs[DET_POINTS_KEY], new_outputs[DET_INDICES_KEY], new_outputs[DET_SCORES_KEY] = \
                self.detect_points_in_heatmap(outputs[POINT_HEATMAP_KEY])

        return new_outputs

    def detect_points_in_heatmap(self, heatmap: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Detects a set of points in a heatmap using the blob detector.

        Heatmap expected in the shape of: (Nc, H, W).
        If ignore index is specified, no points are attempted to be detected in that channel of the
        heatmap.

        Returns the detected points, class indices and scores, sorted in descending order by score.
        """
        if heatmap.ndim != 3:
            raise RuntimeError(f'Heatmap should have 3 dimensions. Has: {heatmap.ndim}')

        points, indices, scores = [], [], []
        for cls_idx, cls_heatmap in enumerate(heatmap):
            # (H, W)
            if cls_idx == self.ignore_index:
                # Don't detect points in this heatmap
                continue

            # Convert to numpy uint8
            hm = (cls_heatmap.detach().cpu().numpy() * 255).astype(np.uint8)

            # Perform blob detection (Getting back a set of keypoints)
            kps = self.detector.detect(hm)

            points.append(torch.as_tensor([k.pt for k in kps]))
            indices.append(torch.as_tensor([cls_idx for _ in range(len(points[-1]))],
                                           dtype=torch.long))

            # Look in the heatmap to find the class 'score'
            hm_scores = []
            for pt in points[-1]:
                hm_scores.append(cls_heatmap[int(pt[1]), int(pt[0])])
            scores.append(torch.as_tensor(hm_scores))

        # Collect into single tensors
        points, indices, scores = torch.cat(points, dim=0), torch.cat(indices, dim=0), torch.cat(scores, dim=0)

        # Sort by scores in descending order
        scores, sorted_indices = torch.sort(scores, descending=True)
        points = points[sorted_indices]
        indices = indices[sorted_indices]

        return points, indices, scores
