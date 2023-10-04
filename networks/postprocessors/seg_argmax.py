from typing import List, Dict, Any

import torch

from networks.postprocessors.postprocessor import Postprocessor
from util.constants import SEG_MASK_LOGITS_KEY, SEG_MASK_INT_KEY


class SegArgmax(Postprocessor):
    def __init__(self):
        """Creates a postprocessor for taking the argmax of segmentation mask logits.
        """
        super().__init__()

    @property
    def model_output_keys(self) -> List[str]:
        return [SEG_MASK_LOGITS_KEY]

    def postprocess(self, outputs: Dict[str, Any]) -> Dict[str, Any]:
        mask_logits = outputs[SEG_MASK_LOGITS_KEY]
        mask_int = torch.argmax(mask_logits, dim=-3).to(torch.uint8)

        return {**outputs, SEG_MASK_INT_KEY: mask_int}
