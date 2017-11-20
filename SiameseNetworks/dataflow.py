
import os, sys
from collections import defaultdict

import numpy as np
import cv2

from torch.utils.data import Dataset

# Setup common_utils package:
root_path = os.path.abspath(os.path.join(".."))
if not root_path in sys.path:
    sys.path.append(root_path)

from common_utils.dataflow import ProxyDataset, TransformedDataset


class OmniglotDataset(Dataset):
    """
    Class represents Omniglot dataset in pytorch framework
    """
    def __init__(self, dataset_type="Train", data_path="", alphabet_char_id_drawers_ids={}, drawers_ids=None,
                 shuffle=True):
        assert dataset_type in ["Train", "Test"]
        assert len(alphabet_char_id_drawers_ids) > 0
        super(OmniglotDataset, self).__init__()
        self.dataset_type = dataset_type
        self.data_path = data_path
        self.alphabet_char_id_drawers_ids = alphabet_char_id_drawers_ids
        self.drawers_ids = drawers_ids
        self.shuffle = shuffle

        if drawers_ids is None:
            drawer_cond_fn = lambda _id: True
        else:
            drawer_cond_fn = lambda _id: str(_id[-3:]) in drawers_ids

        self.data_ids = []
        for a in alphabet_char_id_drawers_ids:
            alphabet_char_ids = alphabet_char_id_drawers_ids[a]
            for char_id in alphabet_char_ids:
                p = os.path.join(a, char_id)
                self.data_ids.extend([os.path.join(p, "%s.png" % _id) for _id in alphabet_char_ids[char_id]
                                      if drawer_cond_fn(_id)])
        self.data_ids = np.array(self.data_ids)
        if self.shuffle:
            np.random.shuffle(self.data_ids)

    def _get_image(self, image_id):
        path = os.path.join(self.data_path, image_id)
        assert os.path.exists(path), "Path '%s' does not exist" % path
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return np.expand_dims(img, axis=-1)

    def _get_label(self, image_id):
        # Remove .png and char id from the image_id
        return os.path.dirname(image_id)

    def __len__(self):
        return len(self.data_ids)

    def __getitem__(self, index):
        return self._get_image(self.data_ids[index]), self._get_label(self.data_ids[index])


def _create_same_pairs(labels_indices, nb_samples_per_class):
    same_pairs = []
    for indices in labels_indices.values():
        same_pairs.extend([np.random.choice(indices, size=2, replace=False) for _ in range(nb_samples_per_class)])
    return np.array(same_pairs)


def _create_diff_pairs(labels_indices, nb_samples_per_two_classes):
    diff_pairs = []
    for i, indices1 in enumerate(labels_indices.values()):
        for j, indices2 in enumerate(labels_indices.values()):
            if (i <= j):
                continue
            ind1 = np.random.choice(indices1, size=nb_samples_per_two_classes)
            ind2 = np.random.choice(indices2, size=nb_samples_per_two_classes)
            diff_pairs.extend([[_i, _j] for _i, _j in zip(ind1, ind2)])
    return np.array(diff_pairs)


class SameOrDifferentPairsDataset(ProxyDataset):
    """
    Create a dataset of pairs uniformly sampled from input dataset
    Pairs are set of two images classified as
        - 'same' if images are from the same class
        - 'different' if images are from different classes
    """

    def __init__(self, ds, nb_pairs, shuffle=True):
        super(SameOrDifferentPairsDataset, self).__init__(ds)
        self.nb_pairs = nb_pairs
        # get mapping y_label -> indices
        labels_indices = defaultdict(list)
        alphabet_indices = defaultdict(list)
        for i, (_, y) in enumerate(ds):
            alphabet_indices[y].append(i)
            y = y.split("/")[0]
            labels_indices[y].append(i)

        half_nb_pairs = int(nb_pairs // 2)
        n1 = int(np.ceil(half_nb_pairs / len(alphabet_indices)))
        same_pairs = _create_same_pairs(alphabet_indices, n1)
        if len(same_pairs) > half_nb_pairs:
            same_pairs = same_pairs[:half_nb_pairs, :]

        n2 = int(np.ceil(nb_pairs / (len(labels_indices) * (len(labels_indices) - 1))))
        diff_pairs = _create_diff_pairs(labels_indices, n2)
        if len(diff_pairs) > half_nb_pairs:
            diff_pairs = diff_pairs[:half_nb_pairs, :]

        self.pairs = np.concatenate((same_pairs, diff_pairs), axis=0)
        if shuffle:
            np.random.shuffle(self.pairs)

    def __len__(self):
        return self.nb_pairs

    def __getitem__(self, index):
        i1, i2 = self.pairs[index, :]
        x1, y1 = self.ds[i1]
        x2, y2 = self.ds[i2]
        return x1, x2, int(y1 == y2)


class PairTransformedDataset(TransformedDataset):
    def __getitem__(self, index):
        x1, x2, y = self.ds[index]
        x1 = self.x_transforms(x1)
        x2 = self.x_transforms(x2)
        if self.y_transforms is not None:
            y = self.y_transforms(y)
        return x1, x2, y