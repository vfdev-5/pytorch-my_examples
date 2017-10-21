from collections import defaultdict, Hashable

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import DataLoaderIter

class ProxyDataset(Dataset):
    
    def __init__(self, ds):
        assert isinstance(ds, Dataset)
        self.ds = ds        

    def __len__(self):
        return len(self.ds)
    

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

    
class OnCudaDataLoaderIter(DataLoaderIter):
    
    def __next__(self):
        batch = super(OnCudaDataLoaderIter, self).__next__()
        cuda_batch = []
        for b in batch:
            if not b.is_pinned():
                b = b.pin_memory()
            cuda_batch.append(b.cuda(async=True))            
        return cuda_batch
    
    
class OnCudaDataLoader(DataLoader):
        
    def __iter__(self):
        return OnCudaDataLoaderIter(self)
    

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
