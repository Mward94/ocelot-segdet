"""Computes metrics related to the task of point detection.

Detection metrics are computed based on the Euclidean distance between the detected and ground
truth points. Distance thresholds set in terms of microns.

If ignore_class is specified, detections and ground truths belonging to that class are not
accumulated, making them excluded from all metrics.
"""
from itertools import chain
from typing import Sequence, Optional, List, Dict, Any

import numpy as np

from networks.metrics.rafaelpadilla.bounding_box import BoundingBox
from networks.metrics.rafaelpadilla.coco_evaluator import get_coco_metrics_points
from networks.metrics.rafaelpadilla.enumerators import BBType, BBFormat
from util.constants import (
    DET_POINTS_KEY, DET_INDICES_KEY, DET_SCORES_KEY, GT_POINTS_KEY, GT_INDICES_KEY, INPUT_MPP_KEY)


class PointDetMetrics:
    """Computes point-based detection metrics.
    """
    def __init__(
            self,
            micron_thresholds: Sequence[float] = (3, ),
            max_dets: Optional[int] = None,
            class_list: Optional[List[str]] = None,
            compute_conf_mat_info: bool = True,
            ignore_class: Optional[str] = None,
    ):
        """Computes various metrics for point-based detection.

        max_dets can be used to specify the top-N detections to be considered for these metrics. If
        None, all detections are used.

        A detection is considered a True Positive if it falls within a Euclidean distance < the
        micron threshold. The AP at each micron threshold will be evaluated per-class and
        aggregated across all classes, considering the number of detections per-threshold as
        described by max_dets. The aggregated metric will have the name:
        "AP Euclidean <micron_threshold>microns", whilst the per-class metrics will have names of
        the form: "AP Euclidean <micron_threshold> microns <name>", where <name> is the class name
        (if class_list is specified), otherwise will be "Class <class_idx>" where class_idx is the
        detection classification index.

        Args:
            micron_thresholds: A collection of distance thresholds to compute the per-class and
                aggregated mAP for.
            max_dets: The maximum number of detections to consider when computing metrics. If not
                provided, uses all detections. Default COCO metrics use 100.
            class_list: A list of class names. Indexes of names are used to map indexes of classes.
            compute_conf_mat_info: If set, will also return (per-class), the TP/FP/FN info, along
                with precision/recall/F1.
            ignore_class: Name of a class (as defined in class_list) to ignore from computation.
                Ignored classes are excluded from the averages.
        """
        super().__init__()

        self.max_dets = max_dets
        self.class_list = class_list

        # Validate the ignore class
        if ignore_class is not None:
            if self.class_list is None:
                raise ValueError('Ignore class is specified without the class list')
            if len(self.class_list) == 1:
                raise ValueError('Cannot ignore a class if only 1 class exists in the class list.')
            if ignore_class not in self.class_list:
                raise ValueError(f'Class \'{ignore_class}\' not found in the class list.')

        # Set up the ignore_index
        self.ignore_index = -1 if ignore_class is None else self.class_list.index(ignore_class)

        # Stores for ground truth and predictions
        self.ground_truths = []
        self.predictions = []

        # Keep track of each sample given to update() (Used to associate with a unique image)
        self.sample_idx = 0

        # The main thresholds to evaluate over
        self.micron_thresholds = micron_thresholds

        # Other metrics to compute
        self.compute_conf_mat_info = compute_conf_mat_info

    @property
    def model_output_keys(self) -> List[str]:
        return [DET_POINTS_KEY, DET_INDICES_KEY, DET_SCORES_KEY]

    @property
    def ground_truth_keys(self) -> List[str]:
        return [GT_POINTS_KEY, GT_INDICES_KEY, INPUT_MPP_KEY]

    def update(
            self, outputs: Dict[str, Any], ground_truths: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Updates the state of the metric object given a new set of outputs/ground truths.

        Outputs and ground truths are expected to be passed in as a batch.

        Args:
            outputs: A dictionary containing all model outputs with keys as defined in
                self.model_output_keys.
            ground_truths: A dictionary containing all ground truth data with keys as defined in
                self.ground_truth_keys.
        """
        # Extract prediction data from outputs
        output_points = outputs[DET_POINTS_KEY]
        output_indices = outputs[DET_INDICES_KEY]
        output_confs = outputs[DET_SCORES_KEY]

        # Extract ground truth data from ground_truths
        gt_points = ground_truths[GT_POINTS_KEY]
        gt_indices = ground_truths[GT_INDICES_KEY]

        # Extract MPP information
        image_mpps = ground_truths[INPUT_MPP_KEY]

        # Append all data to store (Note: Each point/ind/conf contains all preds/gts for an input)
        for idx, (out_pt, out_ind, out_conf, gt_pt, gt_ind) in enumerate(zip(
                output_points, output_indices, output_confs, gt_points, gt_indices)):
            # Put all data onto cpu in numpy arrays
            out_pt = out_pt.detach().cpu().numpy()
            out_ind = out_ind.detach().cpu().numpy()
            out_conf = out_conf.detach().cpu().numpy()
            gt_pt = gt_pt.detach().cpu().numpy()
            gt_ind = gt_ind.detach().cpu().numpy()

            # Extract the single images' MPP
            img_mpp = image_mpps[idx]

            # Create (and append) all points for this image.
            # Set image_name as a unique image index, so all points are associated to the same image
            # Given we are using the raphaelpadilla library, we create boxes with 0 width/height
            self.ground_truths.extend([
                BoundingBox(image_name=str(self.sample_idx), class_id=per_pt_gt_ind,
                            coordinates=tuple([*per_pt_gt_coord, *per_pt_gt_coord]),
                            bb_type=BBType.GROUND_TRUTH, format=BBFormat.XYX2Y2, img_mpp=img_mpp)
                for per_pt_gt_ind, per_pt_gt_coord in zip(gt_ind, gt_pt) if per_pt_gt_ind != self.ignore_index
            ])
            self.predictions.extend([
                BoundingBox(image_name=str(self.sample_idx), class_id=per_pt_out_ind,
                            coordinates=tuple([*per_pt_out_coord, *per_pt_out_coord]),
                            confidence=per_pt_out_conf, bb_type=BBType.DETECTED,
                            format=BBFormat.XYX2Y2, img_mpp=img_mpp)
                for per_pt_out_ind, per_pt_out_coord, per_pt_out_conf in zip(
                    out_ind, out_pt, out_conf) if per_pt_out_ind != self.ignore_index
            ])

            # Update the sample index
            self.sample_idx += 1

        return None

    def compute(self) -> Dict[str, Any]:
        """Computes a final set of metrics given all observations made so far.

        If there are no ground truths and no predictions, we return no metrics (an empty
        dictionary).

        Returns:
            A dictionary containing all computed metric values. Keys = name of metric, Values =
                value of metric.
        """
        metrics = {}
        if len(self.ground_truths) == len(self.predictions) == 0:
            return metrics

        # ### Compute metrics ###
        for threshold in self.micron_thresholds:
            # Create a string representation of this threshold
            threshold_str = f'{threshold:.2f}'
            if threshold_str.endswith('00'):
                threshold_str = f'{threshold:.0f}'
            threshold_str = f'{threshold_str} microns'

            # ##################### Get metrics accumulated across all images  #####################
            # Extract the MPP of the image associated to each ground truth point
            # Currently only a single MPP across all images is supported
            # If there are no ground truth points, find the MPP from a predicted point
            image_mpps = {tuple(pt.img_mpp) for pt in chain(self.ground_truths, self.predictions)}
            if len(image_mpps) != 1:
                raise NotImplementedError(
                    f'Finding Euclidean distance based on microns when different images have '
                    f'different mpps is currently unsupported. MPPs: {image_mpps}')
            image_mpp = list(image_mpps)[0]

            # Only support matching MPP in both axes
            if image_mpp[0] != image_mpp[1]:
                raise NotImplementedError(
                    'Different X/Y MPPs unsupported for finding Euclidean distance based on micron '
                    'thresholds.')

            # Convert the micron threshold to pixels through image MPP
            px_threshold = threshold / image_mpp[0]

            coco_metrics = get_coco_metrics_points(
                self.ground_truths, self.predictions, distance_threshold=px_threshold,
                max_dets=self.max_dets)

            # ##################################### TP/FP data #####################################
            if self.compute_conf_mat_info:
                metrics.update(self.get_conf_mat_info(coco_metrics, f'Euclidean {threshold_str}'))

        # Return computed metrics
        return metrics

    def reset(self):
        """Resets any internal stores so that metric instances can be reused

        Clears the bounding box stores and resets the sample idx
        """
        self.ground_truths.clear()
        self.predictions.clear()
        self.sample_idx = 0

    def get_conf_mat_info(self, metrics: Dict[str, Any], metric_str: str) -> Dict[str, float]:
        """Computes the TP/FP/FN information, including precision/recall/f1.

        Macro metrics are computed by taking the average across all classes.

        Args:
            metrics: The computed metrics from the rafaelpadilla library.
            metric_str: The base name to use for the metric.

        Returns:
            A dictionary mapping metric names to metric values.
        """
        # Store for final metrics
        extracted_metrics = {}

        # Per-class, extract the metrics for that class
        all_precis, all_recall, all_f1, all_np = [], [], [], []
        for class_idx, class_metrics in metrics.items():
            class_name = self.class_name_from_class_idx(class_idx)

            # Extract the TP/FP/FN at the current threshold for this class
            NP = class_metrics['total positives']
            TP = class_metrics['TP']
            FP = class_metrics['FP']
            FN = NP - TP if NP is not None and TP is not None else None
            num_positives = 0 if NP is None else NP

            # Compute precision/recall/f1 at the current threshold for this class
            precis = TP / (TP + FP + 1e-10) if TP is not None and FP is not None else None
            recall = TP / (TP + FN + 1e-10) if TP is not None and FN is not None else None
            f1 = (2 * precis * recall) / (precis + recall + 1e-10) if precis is not None and recall is not None else None

            # Update dictionary with statistics
            extracted_metrics[f'TP {metric_str} {class_name}'] = TP
            extracted_metrics[f'FP {metric_str} {class_name}'] = FP
            extracted_metrics[f'FN {metric_str} {class_name}'] = FN
            extracted_metrics[f'Num Positives {metric_str} {class_name}'] = num_positives
            extracted_metrics[f'Precision {metric_str} {class_name}'] = precis
            extracted_metrics[f'Recall {metric_str} {class_name}'] = recall
            extracted_metrics[f'F1 {metric_str} {class_name}'] = f1

            # Update running metrics
            all_precis.append(precis)
            all_recall.append(recall)
            all_f1.append(f1)
            all_np.append(num_positives)

        # Update dictionary with macro statistics
        extracted_metrics[f'Precision {metric_str} macro'] = np.mean(
            [p for p in all_precis if p is not None]).item()
        extracted_metrics[f'Recall {metric_str} macro'] = np.mean(
            [r for r in all_recall if r is not None]).item()
        extracted_metrics[f'F1 {metric_str} macro'] = np.mean(
            [f for f in all_f1 if f is not None]).item()
        extracted_metrics[f'Num Positives {metric_str} macro'] = np.sum(all_np).item()

        return extracted_metrics

    def class_name_from_class_idx(self, class_idx):
        return self.class_list[class_idx] if self.class_list is not None else f'Class {class_idx}'
