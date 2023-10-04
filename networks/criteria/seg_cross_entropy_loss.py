from typing import Optional, List, Dict

import torch
from torch.nn.functional import cross_entropy

from util.constants import SEG_MASK_LOGITS_KEY, GT_SEG_MASK_KEY


class SegCrossEntropyLoss:
    """Cross entropy criterion for segmentation.
    """
    def __init__(
        self,
        ignore_class: Optional[str] = None,
        class_list: Optional[List[str]] = None,
        weight: Optional[List[float]] = None,
    ):
        """

        Args:
            ignore_class: The name of the class to ignore from the loss.
            class_list: A list of class names, where list indices correspond to the model output.
                Only required if also specifying `ignore_class`.
            weight: A list of weights used to assign higher weight to the loss of certain classes.
        """
        super().__init__()

        self.class_list = class_list
        if ignore_class is not None and self.class_list is None:
            raise ValueError('Ignore class is specified without the class list')

        # Set up the class index to ignore when calculating loss.
        if ignore_class is None:
            self.ignore_index = -100
        else:
            self.ignore_index = self.class_list.index(ignore_class)

        # Set up the class weights.
        if weight is not None:
            weight = torch.as_tensor(weight)
        self.weight = weight

    @property
    def model_output_keys(self) -> List[str]:
        return [SEG_MASK_LOGITS_KEY]

    @property
    def ground_truth_keys(self) -> List[str]:
        return [GT_SEG_MASK_KEY]

    def compute(self, outputs: Dict, ground_truths: Dict) -> Dict[str, torch.Tensor]:
        pred_logits = outputs[SEG_MASK_LOGITS_KEY]
        target = ground_truths[GT_SEG_MASK_KEY].long()

        if torch.all(target == self.ignore_index):
            # Produce zero loss when all pixels are ignored.
            # Failing to handle this case results in NaN loss from cross_entropy.
            loss = 0 * pred_logits.sum()
        else:
            # Move class weights to the same device as `outputs`.
            if self.weight is not None:
                self.weight = self.weight.to(pred_logits.device)

            # Calculate the cross entropy loss.
            loss = cross_entropy(pred_logits, target, self.weight, ignore_index=self.ignore_index)

        # Return the loss.
        return {'cross_entropy_loss': loss}
