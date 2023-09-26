"""Collection of PyTorch utility functions.
"""
import random

import numpy as np
import torch


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
