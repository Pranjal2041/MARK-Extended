import h5py
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import numpy as np
from PIL import Image


class CIFARDataset(Dataset):

    def __init__(self, file = 'datasets/cifar100_train.h5', isTrain = True):
        super(CIFARDataset, self).__init__()
        self.data = h5py.File(file, 'r')
        self.set_task(0)
        self.isTrain = isTrain

        # TODO: Understand why exactly this is needed
        self.transformation = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.transformation_feats = transforms.Compose([transforms.Resize(32),
                                        transforms.CenterCrop(28), 
                                        transforms.ToTensor(), 
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

                
    def set_task(self, task_id):
        self.task_id = task_id
        self.X = self.data[str(task_id)]['X']
        self.Y = self.data[str(task_id)]['Y']
        self.n  = self.X.shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # TODO: Prefer transpose in original dataset
        img = Image.fromarray(self.X[idx].transpose(1,2,0).astype(np.uint8))
        return self.transformation(img), self.Y[idx], self.transformation_feats(img)


def get_cifar100_dataloader(fol = 'datasets', isTrain = True, isValid = False, batch_size = 16, num_workers = 4):
    dataset = CIFARDataset(os.path.join(fol, f'cifar100_{"train" if isTrain else "val" if isValid else "test"}.h5'), isTrain)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = batch_size, shuffle = isTrain, num_workers = num_workers)
    return dataloader

if __name__ == '__main__':
    # Pass isTrain and isValid to suitably get the correct split
    dl = get_cifar100_dataloader(num_workers= 0) # Train dl
    test_dl = get_cifar100_dataloader(isTrain = False, num_workers= 0) # Test dl
    valid_dl = get_cifar100_dataloader(isTrain = False, isValid = True, num_workers= 0) # Valid dl
    print(f'Current Task Id: {dl.dataset.task_id}')
    # There are 3 outputs, see original code for why 3 variables were returned instead of 2
    for i, (img, label, img_feats) in enumerate(dl):
        print(f'{i} - {label}')
        print(img.shape)
        print(img_feats.shape)
        if i == 3:
            break

    # Switch to second task, initially task is 0
    dl.dataset.set_task(2)
    print(f'Current Task Id: {dl.dataset.task_id}')
    for i, (img, label, img_feats) in enumerate(dl):
        print(f'{i} - {label}')
        print(img.shape)
        print(img_feats.shape)
        if i == 3:
            break



