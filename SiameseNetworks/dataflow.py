
import os, sys
from collections import defaultdict

import numpy as np
import cv2

import torch
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
            if i <= j:
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

    def __init__(self, ds, nb_pairs, class_indices=None, shuffle=True, seed=None):
        super(SameOrDifferentPairsDataset, self).__init__(ds)
        self.nb_pairs = nb_pairs

        if class_indices is None:
            # get mapping y_label -> indices
            class_indices = defaultdict(list)
            for i, (_, y) in enumerate(ds):
                class_indices[y].append(i)

        if shuffle and seed is not None:
            np.random.seed(seed)

        half_nb_pairs = int(nb_pairs // 2)
        self.nb_same_pairs_per_class = int(np.ceil(half_nb_pairs / len(class_indices)))
        self.same_pairs = _create_same_pairs(class_indices, self.nb_same_pairs_per_class)
        if len(self.same_pairs) > half_nb_pairs:
            if shuffle:
                np.random.shuffle(self.same_pairs)
            self.same_pairs = self.same_pairs[:half_nb_pairs, :]

        self.nb_samples_per_two_classes = int(np.ceil(nb_pairs / (len(class_indices) * (len(class_indices) - 1))))
        self.diff_pairs = _create_diff_pairs(class_indices, self.nb_samples_per_two_classes)
        if len(self.diff_pairs) > half_nb_pairs:
            if shuffle:
                np.random.shuffle(self.diff_pairs)
            self.diff_pairs = self.diff_pairs[:half_nb_pairs, :]

        # self.pairs = np.concatenate((self.same_pairs, self.diff_pairs), axis=0)
        self.pairs = np.zeros((len(self.same_pairs) + len(self.diff_pairs), 2), dtype=np.int)
        for i, (s, d) in enumerate(zip(self.same_pairs, self.diff_pairs)):
            self.pairs[2 * i, :] = s
            self.pairs[2 * i + 1, :] = d

        if shuffle:
            np.random.shuffle(self.pairs)

    def __len__(self):
        return self.nb_pairs

    def __getitem__(self, index):
        i1, i2 = self.pairs[index, :]
        x1, y1 = self.ds[i1]
        x2, y2 = self.ds[i2]
        return [x1, x2], int(y1 == y2)


class SameOrDifferentPairsBatchDataset(ProxyDataset):
    """
    Create a dataset of pairs uniformly sampled from input dataset
    Pairs are set of two images classified as
        - 'same' if images are from the same class
        - 'different' if images are from different classes
    """

    def __init__(self, ds, nb_batches, batch_size,
                 class_indices=None,
                 x_transforms=None, y_transforms=None,
                 pin_memory=True, on_gpu=True):
        super(SameOrDifferentPairsBatchDataset, self).__init__(ds)
        self.nb_batches = nb_batches
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        self.on_gpu = on_gpu
        self.x_transforms = x_transforms if x_transforms is not None else lambda x: x
        self.y_transforms = y_transforms if y_transforms is not None else lambda y: y

        if class_indices is None:
            # get mapping y_label -> indices
            class_indices = defaultdict(list)
            for i, (_, y) in enumerate(ds):
                class_indices[y].append(i)

        self.class_indices = class_indices
        self.classes = list(self.class_indices.keys())

    def __len__(self):
        return self.nb_batches

    def __getitem__(self, index):

        if index >= self.nb_batches:
            raise IndexError()

        random_classes = np.random.choice(self.classes, size=(self.batch_size, ), replace=False)

        targets = np.zeros((self.batch_size, 1), dtype=np.float32)
        # float target is needed in BCEWithLogitsLoss (v 0.2)
        # target should have size (batch_size, 1) same size is produced by network
        targets[self.batch_size // 2:] = 1.0
        targets = torch.from_numpy(targets)

        xs1 = []
        xs2 = []

        for i in range(self.batch_size):
            c = random_classes[i]
            n_samples = len(self.class_indices[c])
            index1 = np.random.randint(0, n_samples)
            x1, _ = self.ds[self.class_indices[c][index1]]
            x1 = self.x_transforms(x1)
            xs1.append(x1.unsqueeze(0))

            if i < self.batch_size // 2:
                diff_classes = list(self.classes)
                diff_classes.remove(c)
                c = diff_classes[np.random.randint(len(diff_classes))]
            n_samples = len(self.class_indices[c])
            index2 = np.random.randint(0, n_samples)
            x2, _ = self.ds[self.class_indices[c][index2]]
            x2 = self.x_transforms(x2)
            xs2.append(x2.unsqueeze(0))

        xs1 = torch.cat(xs1)
        xs2 = torch.cat(xs2)

        if self.pin_memory:
            xs1 = xs1.pin_memory()
            xs2 = xs2.pin_memory()
            targets = targets.pin_memory()
            if self.on_gpu:
                xs1 = xs1.cuda()
                xs2 = xs2.cuda()
                targets = targets.cuda()

        return [xs1, xs2], targets


class PairTransformedDataset(TransformedDataset):
    def __getitem__(self, index):
        (x1, x2), y = self.ds[index]
        x1 = self.x_transforms(x1)
        x2 = self.x_transforms(x2)
        if self.y_transforms is not None:
            y = self.y_transforms(y)
        return [x1, x2], y


