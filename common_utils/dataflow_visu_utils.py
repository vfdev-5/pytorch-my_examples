
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


def display_basic_dataset(ds, max_datapoints=15, n_cols=5, figsize=(12, 6)):
    """
    Method to display datapoints from dataset
    Datapoint is (x, y) and y should be integer

    :param ds: basic dataset without batching
    :param max_datapoints: number of max datapoints to display
    :param n_cols: number of images per line
    :param figsize: figure size
    :return:
    """
    for i, (x, y) in enumerate(ds):

        if i % n_cols == 0:
            plt.figure(figsize=figsize)

        x = _to_ndarray(x)
        y = _to_str(y)

        plt.subplot(1, n_cols, (i % n_cols) + 1)
        plt.imshow(scale_percentile(x, q_min=0.0, q_max=100.0))
        plt.title("Class {}".format(y))

        max_datapoints -= 1
        if max_datapoints == 0:
            break


def display_data_augmentations(ds, aug_ds, max_datapoints=15, n_cols=5, figsize=(12, 6)):
    """
    Method to display two datasets: ordinary and augmented
    :param ds:
    :param aug_ds:
    :param max_datapoints:
    :param n_cols: number of images per line
    :param figsize: figure size
    :return:
    """
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(ds, aug_ds)):

        if i % n_cols == 0:
            plt.figure(figsize=figsize)

        x1 = _to_ndarray(x1)
        x2 = _to_ndarray(x2)
        y1 = _to_str(y1)
        y2 = _to_str(y2)

        plt.subplot(2, n_cols, (i % n_cols) + 1)
        plt.imshow(scale_percentile(x1, q_min=0.0, q_max=100.0))
        plt.title("Orig. Class {}".format(y1))

        plt.subplot(2, n_cols, (i % n_cols) + 1 + n_cols)
        plt.imshow(scale_percentile(x2, q_min=0.0, q_max=100.0))
        plt.title("Aug. Class {}".format(y2))

        max_datapoints -= 1
        if max_datapoints == 0:
            break


def display_batches(batches_ds, max_batches=3, n_cols=5, figsize=(16, 6), suptitle_prefix=""):
    """
    Method to display batches
    :param batches_ds:
    :param max_batches: number of batches to display
    :param n_cols: number of images per line
    :param figsize: figure size
    :param suptitle_prefix: prefix string to add to displayed string "Batch k"
    :return:
    """
    for i, (batch_x, batch_y) in enumerate(batches_ds):

        plt.figure(figsize=figsize)
        plt.suptitle(suptitle_prefix + "Batch %i" % i)
        for j in range(len(batch_x)):
            if j > 0 and j % n_cols == 0:
                plt.figure(figsize=figsize)

            x = batch_x[j, ...]
            y = batch_y[j, ...]

            x = _to_ndarray(x)
            y = _to_str(y)

            plt.subplot(1, n_cols, (j % n_cols) + 1)
            plt.imshow(scale_percentile(x))
            plt.title("Class {}".format(y))

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

    if len(x.shape) == 3 and x.shape[2] == 1:
        x = x[:, :, 0]
    return x


def _to_str(y, width=15):
    if torch.is_tensor(y):
        if y.is_cuda:
            y = y.cpu()
        y = y.numpy()
    y_str = "{}".format(y)
    if len(y_str) < width:
        return y_str
    ret = []
    for i in range(0, len(y_str), width):
        ret.append(y_str[i:i+width])
        ret.append("\n")
    return "".join(ret[:-1])
