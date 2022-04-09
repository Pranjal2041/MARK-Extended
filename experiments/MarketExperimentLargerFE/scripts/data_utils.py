import torch
from torch.utils.data import Dataset

class ImageGridData: ...
class ImageData: ...

class MyDS(Dataset):
    def __init__(self, X,y):
        self.samples = torch.Tensor(X)
        self.labels = torch.LongTensor(y)        
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return (self.samples[idx],self.labels[idx])