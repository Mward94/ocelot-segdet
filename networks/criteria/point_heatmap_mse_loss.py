from typing import Optional, List, Dict, Any

import torch
from torch.nn.functional import mse_loss

from util.constants import SEG_MASK_LOGITS_KEY, GT_POINT_HEATMAP_KEY


class PointHeatmapMSELoss:
    """Cross entropy criterion for segmentation.
    """
    def __init__(self, ignore_class: Optional[str] = None, class_list: Optional[List[str]] = None,
                 class_weights: Optional[List[float]] = None):
        """Computes the MSE loss between a predicted and ground truth point heatmap.

        The predicted heatmap is expected to contain logits from the segmentation model. As such, a
        sigmoid will first be applied to bound values to [0, 1] before computing the loss.

        Args:
            ignore_class: The name of the class to ignore from the loss.
            class_list: A list of class names, where list indices correspond to the model output.
                Only required if also specifying `ignore_class`.
            class_weights: A weighting that is applied PER-CHANNEL to the MSE Loss. For
                completeness, the number of classes specified should be the entire class list (even
                if ignoring a class from the loss).
        """
        super().__init__()

        self.class_list = class_list
        if ignore_class is not None and self.class_list is None:
            raise ValueError('Ignore class is specified without the class list')

        # Set up the class indices to keep when calculating loss.
        if ignore_class is None:
            self.keep_indices = None
        else:
            ignore_index = self.class_list.index(ignore_class)
            self.keep_indices = [i for i in range(len(self.class_list)) if i != ignore_index]

        # Set up the weights for MSE loss.
        if class_weights is not None:
            # This puts the weights in a tensor with shape: [1, C, 1, 1]. To match the batched
            # tensor shape. Handles ignoring the ignore class if required.
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
            if self.keep_indices is not None:
                self.class_weights = self.class_weights[self.keep_indices]
            self.class_weights = self.class_weights.reshape(1, -1, 1, 1)
        else:
            self.class_weights = None

    @property
    def model_output_keys(self) -> List[str]:
        return [SEG_MASK_LOGITS_KEY]

    @property
    def ground_truth_keys(self) -> List[str]:
        return [GT_POINT_HEATMAP_KEY]

    def compute(
            self,
            outputs: Dict[str, Any],
            ground_truths: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        pred_logits = outputs[SEG_MASK_LOGITS_KEY]
        target = ground_truths[GT_POINT_HEATMAP_KEY]

        if self.keep_indices is not None:
            # It's batched!
            pred_logits = pred_logits[:, self.keep_indices]
            target = target[:, self.keep_indices]

        # Calculate the MSE loss.
        if self.class_weights is None:
            loss = mse_loss(torch.sigmoid(pred_logits), target)
        else:
            # Move class weights to the same device as `outputs`.
            self.class_weights = self.class_weights.to(pred_logits.device)
            loss = (self.class_weights * (torch.sigmoid(pred_logits) - target) ** 2).mean()

        # Return the loss.
        return {'mse_loss': loss}
