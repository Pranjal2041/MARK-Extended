import h5py
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import numpy as np
from PIL import Image
import pickle
# from utils import FeatureExtractor as fe


class CIFARDataset(Dataset):

    def __init__(self, file = 'datasets/cifar100_train.h5', isTrain = True, num_tasks = 20):
        super(CIFARDataset, self).__init__()
        self.num_tasks = num_tasks
        self.data = h5py.File(file, 'r')
        self.set_task(0)
        self.isTrain = isTrain

        # TODO: Understand why exactly this is needed
        self.transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
                
    def set_task(self, task_id):
        self.task_id = task_id
        self.X = self.data[str(task_id)]['X']
        self.Y = self.data[str(task_id)]['Y']
        self.n  = self.X.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # TODO: Prefer transpose in original dataset
        img = Image.fromarray((255*self.X[idx]).transpose(1, 2, 0).astype(np.uint8))
        return self.transformation(img), self.Y[idx]

def get_cifar100_dataloader(fol = 'datasets', dset_type = 'cifar100', isTrain=True, isValid=False, batch_size=16, num_workers=4, num_tasks=20):
    dataset = CIFARDataset(os.path.join(fol, f'{dset_type}_{"train" if isTrain else "val" if isValid else "test"}.h5'), isTrain, num_tasks=num_tasks)
    dataset = CIFARDataset(os.path.join(fol, f'{dset_type}_{"train" if isTrain else "val" if isValid else "test"}.h5'), isTrain, num_tasks=num_tasks)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=isTrain, num_workers=num_workers)
    return dataloader


class MarketDataset(Dataset):

    def __init__(self, file = 'datasets/data_mean_na.pkl', length = 16):
        self.file = file
        self.data = pickle.load(open(file,'rb'))
        self.day = 0
        self.symbol = 0
        self.num_days = len(self.data)
        self.num_symbols = 308
        self.length = length

    def set_day(self, day_num):
        self.day = day_num
    
    def set_symbol(self, symbol):
        self.symbol = symbol

    def __len__(self, ):
        return self.length#len(self.data[self.day][self.symbol])


    # TODO: Should volume be normalized?
    def __getitem__(self, idx, ):
        df = self.data[self.day][self.symbol]
        to_normalize = ['Open', 'High', 'Low', 'Close',
            'Open_prev', 'High_prev', 'Low_prev', 'Close_prev', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
            'RSI_14', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0',]
        try:
            df[to_normalize]/=df.iloc[0]['Close'] # Note that repeated application doesnt change anything
        except Exception as e:
            print('Error in Normalization:',e,idx, self.day, self.symbol)

        ifeats = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends',
            'Open_prev', 'High_prev', 'Low_prev', 'Close_prev', 'Volume_prev',
            'Dividends_prev', 'hurst', 'SMA_10', 'SMA_20', 'SMA_50', 'SMA_200',
            'VOL_SMA_20', 'RSI_14', 'BBL_5_2.0', 'BBM_5_2.0', 'BBU_5_2.0',
            'BBB_5_2.0', 'BBP_5_2.0', 'MACD_12_26_9', 'MACDh_12_26_9',
            'MACDs_12_26_9','sym']
        inp_feats = df[ifeats]
        lfeats = ['(0.02, 0.01)', '(0.01, 0.005)', '(0.01, 0.02)',
            '(0.005, 0.01)']
        labels = df[lfeats]

        gap = len(df)/self.length
        low = int(idx*gap)
        high = min(len(df), int((idx+1)*gap))
        if low>=high:
            new_idx = min(low, high)
        else:
            new_idx = np.random.randint(low, high)

        padding = torch.zeros(375 - new_idx - 1, inp_feats.values.shape[1])

        return torch.vstack((torch.from_numpy(inp_feats[:new_idx+1].values),padding)), torch.from_numpy(labels.values[new_idx]), new_idx+1

def get_marketcl_dataloader(fol = 'datasets', batch_size=5, num_workers=0, shuffle = True):
    dataset = MarketDataset(os.path.join(fol, 'data_mean_na.pkl' ))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

if __name__ == '__main__':

    # To get total number of tasks, use dl.dataset.num_tasks
    # To set to a particular task, simply run dl.dataset.set_task(task_id)
    # Now use the dataloader as it is and you will get *batches* of a task


    # Pass isTrain and isValid to suitably get the correct split
    train_dl = get_cifar100_dataloader(num_workers=0) # Train dl
    test_dl = get_cifar100_dataloader(isTrain=False, num_workers=0) # Test dl
    valid_dl = get_cifar100_dataloader(isTrain=False, isValid=True, num_workers=0) # Valid dl
