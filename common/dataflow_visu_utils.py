
try:
    from itertools import izip as zip
except ImportError:
    # will be 3.x series
    pass

import numpy as np
import matplotlib.pylab as plt

from PIL.Image import Image

import torch

from .image_utils import scale_percentile


def display_basic_dataset(ds, max_datapoints=15, n_cols=5):
    """
    Method to display datapoints from dataset
    Datapoint is (x, y) and y should be integer

    :param ds: basic dataset without batching
    :param max_datapoints: number of max datapoints to display
    :return:
    """
    for i, (x, y) in enumerate(ds):

        if i % n_cols == 0:
            plt.figure(figsize=(12, 4))

        x = _to_ndarray(x)

        plt.subplot(1, n_cols, (i % n_cols) + 1)
        plt.imshow(scale_percentile(x, q_min=0.0, q_max=100.0))
        plt.title("Class %i" % y)

        max_datapoints -= 1
        if max_datapoints == 0:
            break


def display_data_augmentations(ds, aug_ds, max_datapoints=15, n_cols=5):
    """
    Method to display two datasets: ordinary and augmented
    :param ds:
    :param aug_ds:
    :param max_datapoints:
    :param n_cols: number of images per row
    :return:
    """
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(ds, aug_ds)):

        if i % n_cols == 0:
            plt.figure(figsize=(12, 6))

        x1 = _to_ndarray(x1)
        x2 = _to_ndarray(x2)

        plt.subplot(2, n_cols, (i % n_cols) + 1)
        plt.imshow(scale_percentile(x1, q_min=0.0, q_max=100.0))
        plt.title("Orig. Class %i" % y1)

        plt.subplot(2, n_cols, (i % n_cols) + 1 + n_cols)
        plt.imshow(scale_percentile(x2, q_min=0.0, q_max=100.0))
        plt.title("Aug. Class %i" % y2)

        max_datapoints -= 1
        if max_datapoints == 0:
            break


def display_batches(batches_ds, max_batches=3, n_cols=5):

    for i, (batch_x, batch_y) in enumerate(batches_ds):

        plt.figure(figsize=(12, 4))
        plt.suptitle("Batch %i" % i)
        for j in range(len(batch_x)):
            if j > 0 and j % n_cols == 0:
                plt.figure(figsize=(16, 4))

            x = batch_x[j, ...]
            y = batch_y[j, ...]

            x = _to_ndarray(x)

            plt.subplot(1, n_cols, (j % n_cols) + 1)
            plt.imshow(scale_percentile(x))
            plt.title("Class %i" % y)

        max_batches -= 1
        if max_batches == 0:
            break


def _to_ndarray(x):
    if torch.is_tensor(x):
        if x.is_cuda:
            x = x.cpu()
        x = x.numpy().transpose([1, 2, 0])
    elif isinstance(x, Image):
        x = np.asarray(x)
    assert isinstance(x, np.ndarray), "x is of type {}".format(type(x))
    return x
