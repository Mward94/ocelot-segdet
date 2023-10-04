"""Postprocessor to apply Gaussian modulation to a point heatmap.

This follows the method described here:
https://openaccess.thecvf.com/content_CVPR_2020/papers/Zhang_Distribution-Aware_Coordinate_Representation_for_Human_Pose_Estimation_CVPR_2020_paper.pdf

The kernel size used is 6x the standard deviation + 1 (to encompass the mean and 3x stddevs).
"""
import math
from typing import List, Dict, Any

import torch
from torchvision.transforms.functional import gaussian_blur

from networks.postprocessors.postprocessor import Postprocessor
from util.constants import POINT_HEATMAP_KEY


class GaussianModulation(Postprocessor):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma
        self.kernel_size = 6 * math.ceil(self.sigma) + 1

    @property
    def model_output_keys(self) -> List[str]:
        return [POINT_HEATMAP_KEY]

    def postprocess(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        heatmap = outputs[POINT_HEATMAP_KEY]

        # Apply the Gaussian
        blurred_heatmap = gaussian_blur(heatmap, kernel_size=[self.kernel_size, self.kernel_size],
                                        sigma=self.sigma)

        # Create the scaled heatmap and initialize appropriately
        scaled_heatmap = torch.zeros_like(blurred_heatmap)

        # Per-channel, scale the blurred heatmap back as described in the paper
        if blurred_heatmap.ndim == 4:       # Batched
            for b_idx in range(scaled_heatmap.shape[0]):
                for c_idx in range(scaled_heatmap.shape[1]):
                    scaled_heatmap[b_idx, c_idx] = scale_heatmap(blurred_heatmap[b_idx, c_idx], heatmap[b_idx, c_idx])
        else:
            for c_idx in range(scaled_heatmap.shape[0]):
                scaled_heatmap[c_idx] = scale_heatmap(blurred_heatmap[c_idx], heatmap[c_idx])

        return {
            **outputs,
            POINT_HEATMAP_KEY: scaled_heatmap,
        }


def scale_heatmap(blurred_heatmap, original_heatmap):
    return original_heatmap.max() * ((blurred_heatmap - blurred_heatmap.min()) /
                                     (blurred_heatmap.max() - blurred_heatmap.min()))
