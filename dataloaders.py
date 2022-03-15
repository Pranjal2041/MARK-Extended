import h5py
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import os
import numpy as np
from PIL import Image
from utils import FeatureExtractor as fe


class CIFARDataset(Dataset):

    def __init__(self, file = 'datasets/cifar100_train.h5', isTrain = True, num_tasks = 20):
        super(CIFARDataset, self).__init__()
        self.num_tasks = num_tasks
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
        img = Image.fromarray(self.X[idx].transpose(1, 2, 0).astype(np.uint8))
        return self.transformation(img), self.Y[idx], self.transformation_feats(img)


def get_cifar100_dataloader(fol = 'datasets', isTrain=True, isValid=False, batch_size=16, num_workers=4, num_tasks=20):
    dataset = CIFARDataset(os.path.join(fol, f'cifar100_{"train" if isTrain else "val" if isValid else "test"}.h5'), isTrain, num_tasks=num_tasks)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=isTrain, num_workers=num_workers)
    return dataloader

if __name__ == '__main__':

    # To get total number of tasks, use dl.dataset.num_tasks
    # To set to a particular task, simply run dl.dataset.set_task(task_id)
    # Now use the dataloader as it is and you will get *batches* of a task


    # Pass isTrain and isValid to suitably get the correct split
    train_dl = get_cifar100_dataloader(num_workers=0) # Train dl
    test_dl = get_cifar100_dataloader(isTrain=False, num_workers=0) # Test dl
    valid_dl = get_cifar100_dataloader(isTrain=False, isValid=True, num_workers=0) # Valid dl
    # print(f'Current Task Id: {dl.dataset.task_id}')
    # There are 3 outputs, see original code for why 3 variables were returned instead of 2
    net = fe.FeatureExtractor(sample_dim=32, input_size=3, hidden_size=32, output_size=128, lr=0.01)
    num_tasks = train_dl.dataset.num_tasks
    nets = {}
    for task_id in range(num_tasks):
        print(f'Task Id: {task_id}')
        train_dl.dataset.set_task(task_id)
        task_net = fe.TrainFeature(net, train_dl)
        nets[task_id] = task_net

    num_tasks = test_dl.dataset.num_tasks
    for task_id in range(num_tasks):
        print(f'Task Id: {task_id}')
        test_dl.dataset.set_task(task_id)
        embedding = fe.get_feature_embedding(nets[task_id], test_dl)
        print(embedding)
    # nets = fe.TrainFeature(net, dl)
    # for i, (img, label, img_feats) in enumerate(test_dl):
    #     print(f'{i} - {label}')
    #     #print(img.shape)
    #     #print(img_feats.shape)
    #     embedding, _ = fe.get_feature_embedding(nets, i, img)
    #     print(embedding.shape)
    #     print(embedding)
    #     if i == 3:
    #         break

    # # Switch to second task, initially task is 0
    # dl.dataset.set_task(2)
    # print(f'Current Task Id: {dl.dataset.task_id}')
    # for i, (img, label, img_feats) in enumerate(dl):
    #     print(f'{i} - {label}')
    #     print(img.shape)
    #     print(img_feats.shape)
    #     if i == 3:
    #         break



