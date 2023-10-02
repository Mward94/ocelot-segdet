"""Collection of PyTorch utility functions.
"""
import random
from typing import Optional, Mapping

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import _LRScheduler

from util.constants import INPUT_IMAGE_KEY, GT_POINT_HEATMAP_KEY, INPUT_IMAGE_MASK_KEY


def get_default_device() -> torch.device:
    """Gets the default device string.

    This is ``"cuda"`` if CUDA is available, otherwise ``"cpu"``.

    Returns:
        String representing the default device.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)


def move_data_to_device(data, device):
    """Moves data to a given device

    If <data> is a list/tuple, each element in <data> is individually moved to the device, and a
    list of data returned. Otherwise, <data> is moved to the device

    Args:
        data (Tensor/list/tuple): Data to move to the given device
        device (Device): Device to move data to

    Returns:
        data (Tensor/list): Data moved to the given device
    """
    # If any data is a tuple/list, put each element onto the device individually
    if isinstance(data, (tuple, list)):
        return [move_data_to_device(dat, device) for dat in data]
    if not isinstance(data, torch.Tensor):
        return data
    return data.to(device)


def seed_all(seed: int):
    """Seeds all RNGs with a specific seed value.

    Args:
        seed: Seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_current_lr(optimiser):
    """Gets the current Learning Rate from the optimiser.
    """
    return optimiser.param_groups[0]['lr']


def save_model_state(
    target_filepath: str,
    model: nn.Module,
    epoch_num: Optional[int] = None,
    optimiser: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[_LRScheduler] = None,
):
    """Stores the model state to file along with some other fields.

    Args:
        target_filepath: Path to write the saved model state to. By convention it should have
            a .pth extension.
        model: Model to save state of.
        epoch_num: The epoch number that was just trained. This starts at 0
            e.g. If epoch_num = 0, then the model has been trained for 1 epoch.
        optimiser: If given, will also save the state of the optimiser.
        scheduler: If given, will also save the state of the scheduler.
    """
    data = {'state_dict': model.state_dict()}

    if epoch_num is not None:
        data['epoch'] = epoch_num
    if optimiser is not None:
        data['optimiser'] = optimiser.state_dict()
    if scheduler is not None:
        data['scheduler'] = scheduler.state_dict()

    torch.save(data, target_filepath)


def detection_collate_fn(batch, key=None):
    """Custom collate function for detection tasks

    This collates the INPUT_IMAGE_KEY, GT_POINT_HEATMAP_KEY, and INPUT_IMAGE_MASK_KEY as normal.
    All other keys are collated as tuples (given points/indices must be collated this way).
    """
    _NORMAL_COLLATE_KEYS = (INPUT_IMAGE_KEY, GT_POINT_HEATMAP_KEY, INPUT_IMAGE_MASK_KEY)
    elem = batch[0]
    if isinstance(elem, Mapping):
        return {key: detection_collate_fn([d[key] for d in batch], key=key) for key in elem}
    if isinstance(elem, torch.Tensor) and key in _NORMAL_COLLATE_KEYS:
        return torch.stack(batch, 0)
    return tuple(batch)
