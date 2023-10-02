"""Classes to handle accumulating values.
"""
from math import sqrt

import numpy as np
import torch


class MeanValueMeter:
    """A meter for calculating the mean and standard deviation of values."""
    def __init__(self):
        self.n = 0
        self.sum = 0
        self.sum_of_squares = 0
        self.mean = np.nan
        self.stddev = np.nan

    def add(self, value, n=1):
        """Add a value to the meter.

        Args:
            value: the value to add.
            n (int): the number of units represented by the value (default is 1).
        """
        if n <= 0:
            raise ValueError(f'Error. n must be positive. Is: {n}')

        self.sum += value
        self.sum_of_squares += value ** 2
        self.n += n
        self.mean = self.sum / self.n

        if self.n == 1:
            self.stddev = np.inf
        else:
            variance = (self.sum_of_squares - self.n * self.mean ** 2) / (self.n - 1)
            self.stddev = sqrt(max(variance, 0))

    def get_mean(self):
        """Gets the mean of values added to the meter

        The mean value is returned as a python float

        Returns:
            (float): Mean value
        """
        if isinstance(self.mean, torch.Tensor):
            return self.mean.item()
        return self.mean

    def value(self):
        """Get statistics of values added to the meter.

        Returns:
            tuple: the mean and standard deviation of values added to the meter.
        """
        return self.mean, self.stddev

    def reset(self):
        self.n = 0
        self.sum = 0
        self.sum_of_squares = 0
        self.mean = np.nan
        self.stddev = np.nan
