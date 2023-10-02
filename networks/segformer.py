from typing import List, Dict, Any, Union, Tuple

import numpy as np
import segformer.model
import torch
from torch import nn
from torchvision.transforms import functional as F

from util.constants import INPUT_IMAGE_MASK_KEY, INPUT_IMAGE_KEY, SEG_MASK_LOGITS, SEG_MASK_INT
from util.helpers import calculate_cropped_size


class SegFormer(nn.Module):
    def __init__(self, num_classes=3, size='b0', pretrained=False, train_output_crop_margin=0,
                 eval_output_crop_margin=0, input_with_mask: bool = False, mask_channels: int = 1):
        """Creates a SegFormer model using the repo from https://github.com/anibali/segformer

        Args:
            num_classes: Number of classes to predict.
            size: Size of Segformer model (b0 to b5).
            pretrained: Whether pretrained ImageNet weights should be loaded.
            train_output_crop_margin: Crop margin used to center crop the output during training to
                match the required target shape.
            eval_output_crop_margin: Crop margin used to center crop the output during evaluation
                to match the required target shape.
            input_with_mask: If True, expects the input image data to be combined with a mask.
            mask_channels: Number of channels in the mask.
        """
        super().__init__()

        # Instantiate the model
        self.model = getattr(segformer.model, f'segformer_{size}')(pretrained=pretrained, num_classes=num_classes)

        # If passing in a mask with input, modify input layer
        self.input_with_mask = input_with_mask
        if self.input_with_mask:
            # Get number of channels in mask
            self.mask_channels = mask_channels

            # Get reference to existing layer
            input_layer = self.model.backbone.stages[0].patch_embed.proj

            # Create new layer, copy weights where applicable (input channels = 3 (RGB) + mask channels)
            new_input_layer = nn.Conv2d(
                3 + self.mask_channels, input_layer.out_channels,
                kernel_size=input_layer.kernel_size, stride=input_layer.stride,
                padding=input_layer.padding)
            new_input_layer.weight.data[:, :3, ...] = input_layer.weight.data
            new_input_layer.bias.data[:] = input_layer.bias.data

            # Update the model layer
            self.model.backbone.stages[0].patch_embed.proj = new_input_layer

        # Store output crop margin (for training and eval)
        self.train_output_crop_margin = train_output_crop_margin
        self.eval_output_crop_margin = eval_output_crop_margin

    def state_dict(self, *args, **kwargs):
        """Delegates to `torch.nn.Module.state_dict`.
        """
        return self.model.state_dict(*args, **kwargs)

    def load_weights(self, weights_path):
        """Loads a set of weights into the model.
        """
        model_device = next(self.model.parameters()).device
        self.model.load_state_dict(
            torch.load(weights_path, map_location=model_device)['state_dict'])

    @property
    def required_input_keys(self):
        if self.input_with_mask:
            return [INPUT_IMAGE_MASK_KEY]
        return [INPUT_IMAGE_KEY]

    def forward(self, inputs):
        # Do forward pass
        if self.input_with_mask:
            input_images = inputs[INPUT_IMAGE_MASK_KEY]
        else:
            input_images = inputs[INPUT_IMAGE_KEY]
        output = self.model(input_images)
        assert output.shape[-2:] == input_images.shape[-2:]

        # Crop the output if a crop margin is specified
        margin = self.train_output_crop_margin if self.training else self.eval_output_crop_margin
        if margin is not None and margin > 0:
            height, width = calculate_cropped_size(output.shape[-2:], margin)
            output = F.crop(output, top=margin, left=margin, height=height, width=width)

        return {SEG_MASK_LOGITS: output}

    def collate_outputs(
            self, outputs: List[Dict[str, Any]],
            coord_list: List[Union[Tuple[int, int, int, int], List[int]]],
            final_dimensions: Tuple[int, int], background_idx: int = 0
    ) -> Dict[str, Any]:
        """Collates a collection of model outputs into a single output.

        Args:
            outputs: A list of output data, as produced by the DL model.
            coord_list: The coordinates of each output relative to the whole input. Negative
                coordinates can be ignored. Coordinates beyond final_dimensions can also be ignored.
            final_dimensions: The width and height of the region to be reconstructed.
            background_idx: The class index to associate background to (i.e. pixels in seg mask
                where no predictions have been made).

        Returns:
            A single reconstructed output
        """
        _SUPPORTED_KEYS = (SEG_MASK_LOGITS, SEG_MASK_INT)

        # Ensure length of outputs and coord_list match
        if len(outputs) != len(coord_list):
            raise ValueError(f'Must have same number of outputs and coordinates. Have: '
                             f'{len(outputs)} predictions and {len(coord_list)} coordinates')

        # Early exit for a single output (if coordinates are non-negative and within final_dimensions)
        if len(outputs) == 1:
            output_coords = coord_list[0]
            if output_coords[0] >= 0 and output_coords[1] >= 0 and \
                    output_coords[2] <= final_dimensions[0] and output_coords[3] <= \
                    final_dimensions[1]:
                return outputs[0]

        # Ensure at least one of the expected keys are in the outputs
        if not any(key in _SUPPORTED_KEYS for key in outputs[0]):
            raise ValueError(f'Output keys must contain any of {_SUPPORTED_KEYS}. Contains: '
                             f'{outputs[0].keys()}')

        # Create a list of sorted indices by coord_list
        sorted_idxs = [i[0] for i in sorted(enumerate(coord_list), key=lambda x: x[1])]

        # Store for the collated output
        collated_output = {}
        for seg_key in _SUPPORTED_KEYS:
            if seg_key in outputs[0]:
                # Get some properties of the mask data
                mask_w, mask_h = final_dimensions
                mask_dtype = outputs[0][seg_key].dtype
                mask_is_tensor = isinstance(outputs[0][seg_key], torch.Tensor)

                # Set the shape of the mask ((C, H, W) for SEG_MASK, (H, W) for SEG_INT_MASK)
                if seg_key == SEG_MASK_LOGITS:
                    mask_channels = outputs[0][seg_key].shape[0]
                    mask_shape = (mask_channels, mask_h, mask_w)
                    mask_default_value = 0
                elif seg_key == SEG_MASK_INT:
                    mask_shape = (mask_h, mask_w)
                    mask_default_value = background_idx
                else:
                    raise ValueError(f'Key {seg_key} not supported to collate into mask.')

                # Create a mask to collate into
                if mask_is_tensor:
                    seg_mask = mask_default_value * torch.ones(mask_shape, dtype=mask_dtype,
                                                               device=outputs[0][seg_key].device)
                else:
                    seg_mask = mask_default_value * np.ones(mask_shape, dtype=mask_dtype)

                # If a mask of segmentation logits, set them to 1 for the background class
                if seg_key == SEG_MASK_LOGITS:
                    seg_mask[background_idx] = 1

                collated_output[seg_key] = seg_mask

                # Iterate through all outputs/coordinates and collate into the single mask
                for idx in sorted_idxs:
                    output, coord = outputs[idx], coord_list[idx]

                    # Extract the output width and height
                    if seg_key == SEG_MASK_LOGITS:
                        output_h, output_w = output[seg_key].shape[1:]
                    elif seg_key == SEG_MASK_INT:
                        output_h, output_w = output[seg_key].shape
                    else:
                        raise ValueError(f'Key {seg_key} not supported to collate into mask.')

                    # Handle any negative coordinates, or coordinates beyond final_dimensions
                    x1_off = 0 if coord[0] >= 0 else -coord[0]
                    y1_off = 0 if coord[1] >= 0 else -coord[1]
                    x2_off = 0 if coord[2] <= mask_w else coord[2] - mask_w
                    y2_off = 0 if coord[3] <= mask_h else coord[3] - mask_h

                    collated_output[seg_key][..., coord[1] + y1_off:coord[3] - y2_off,
                    coord[0] + x1_off:coord[2] - x2_off] = \
                        output[seg_key][..., y1_off: output_h - y2_off, x1_off:output_w - x2_off]
        return collated_output
