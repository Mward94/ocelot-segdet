"""Postprocessor to turn a segmentation model mask into a point heatmap.

This is done by applying a Sigmoid to the segmentation mask.
"""
from typing import List, Dict, Any

import torch

from networks.postprocessors.postprocessor import Postprocessor
from util.constants import POINT_HEATMAP_KEY, SEG_MASK_LOGITS


class SegMaskToPointHeatmap(Postprocessor):
    def __init__(self):
        super().__init__()

    @property
    def model_output_keys(self) -> List[str]:
        return [SEG_MASK_LOGITS]

    def postprocess(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        return {**outputs, POINT_HEATMAP_KEY: torch.sigmoid(outputs[SEG_MASK_LOGITS])}
