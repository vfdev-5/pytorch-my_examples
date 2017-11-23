from __future__ import print_function

from collections import defaultdict, Hashable

import numpy as np
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import DataLoaderIter


class ProxyDataset(Dataset):

    def __init__(self, ds):
        assert isinstance(ds, Dataset)
        self.ds = ds

    def __len__(self):
        return len(self.ds)


class ResizedDataset(ProxyDataset):

    def __init__(self, ds, output_size, interpolation=cv2.INTER_CUBIC, resize_target=False):
        super(ResizedDataset, self).__init__(ds)
        self.output_size = output_size
        self.interpolation = interpolation
        self.resize_target = resize_target

    def _resize(self, x):
        # RGBA -> RGB
        if x.shape[2] == 4:
            x = x[:, :, 0:3]

        _, _, c = x.shape
        x = cv2.resize(x, dsize=self.output_size, interpolation=self.interpolation)
        if c == 1 and len(x.shape) == 2:
            x = np.expand_dims(x, axis=-1)
        return x

    def __getitem__(self, index):
        x, y = self.ds[index]

        x = self._resize(x)
        if self.resize_target:
            y = self._resize(y)

        return x, y


class CachedDataset(ProxyDataset):

    def __init__(self, ds, n_cached_images=10000):
        super(CachedDataset, self).__init__(ds)
        self.n_cached_images = n_cached_images
        self.cache = {}
        self.cache_hist = []

    def reset(self):
        self.cache = {}

    def __getitem__(self, index):
        if index in self.cache:
            return self.cache[index]
        else:
            x, y = self.ds[index]
            if len(self.cache) > self.n_cached_images:
                first_index = self.cache_hist.pop(0)
                del self.cache[first_index]

            self.cache[index] = (x, y)
            self.cache_hist.append(index)
            return x, y


class TransformedDataset(ProxyDataset):

    def __init__(self, ds, x_transforms, y_transforms=None):
        super(TransformedDataset, self).__init__(ds)
        assert callable(x_transforms)
        if y_transforms is not None:
            assert callable(y_transforms)
        self.ds = ds
        self.x_transforms = x_transforms
        self.y_transforms = y_transforms

    def __getitem__(self, index):
        x, y = self.ds[index]
        x = self.x_transforms(x)
        if self.y_transforms is not None:
            y = self.y_transforms(y)

        return x, y


class OnGPUDataLoaderIter(DataLoaderIter):

    def _to_cuda(self, t):
        if not t.is_pinned():
            t = t.pin_memory()
        return t.cuda(async=True)

    def __next__(self):
        batch = super(OnGPUDataLoaderIter, self).__next__()
        cuda_batch = []
        for b in batch:  # b is (batch_x, batch_y) or ((batch_x1, batch_x2, ...), (batch_y1, batch_y2, ...))
            if torch.is_tensor(b):
                cuda_batch.append(self._to_cuda(b))
            else:
                assert isinstance(b, tuple) or isinstance(b, list)
                cuda_b = []
                for _b in b:
                    assert torch.is_tensor(_b)
                    cuda_b.append(self._to_cuda(_b))
                cuda_batch.append(cuda_b)
        return cuda_batch

    next = __next__  # Python 2 compatibility


class OnGPUDataLoader(DataLoader):

    def __iter__(self):
        return OnGPUDataLoaderIter(self)


class PrintLabelsStats:
    """
    Prints labels stats: 
        - total counts (all previous seen datapoints)
        - counts (current datapoint (e.g. batch))

    The output looks like:
    
    0 | Labels counts: 
        current: | '0': 2 | '1': 4 | '3': 1 | '6': 3 | '7': 4 | '8': 2 | 
        total: | '0': 2 | '1': 4 | '3': 1 | '6': 3 | '7': 4 | '8': 2 | 
   10 | Labels counts: 
        current: | '1': 1 | '2': 3 | '4': 4 | '5': 1 | '6': 2 | '7': 2 | '8': 1 | '9': 2 | 
        total: | '0': 20 | '1': 16 | '2': 22 | '3': 19 | '4': 16 | '5': 15 | '6': 22 | '7': 14 | '8': 18 | '9': 14 | 
    
    """
    
    def __init__(self, ds, display_freq=1, display_total=True):
        assert isinstance(ds, DataLoader)
        assert display_freq > 0, "display_freq should be a positive integer"
        self.ds = ds
        self.total_y_stats = defaultdict(int)
        self.cnt = 0
        self.display_freq = display_freq
        self.display_total = display_total
    
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self):        
        iterator = self.ds.__iter__()
        batch_x, batch_y = next(iterator)
        y_stats = defaultdict(int)
        for dp in batch_y:
            if isinstance(dp, Hashable):
                self.total_y_stats[dp] += 1 
                y_stats[dp] += 1
                    
        if (self.cnt % self.display_freq) == 0:
            print("%i | Labels counts: " % self.cnt)
                                
            print("  current: | ", end='')
            for k in y_stats:
                print("'{}': {} |".format(str(k), y_stats[k]), end=' ')
            print('')
            if self.display_total:
                print("    total: | ", end='')
                for k in self.total_y_stats:
                    print("'{}': {} |".format(str(k), self.total_y_stats[k]), end=' ')
                print('')                    
        self.cnt += 1
        return iterator
