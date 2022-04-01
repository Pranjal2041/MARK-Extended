import torch
import os
from os.path import join
import shutil

def list_to_device(lis, device : str): return [x.to(device) for x in lis]

def get_num_params_in_model(model : torch.nn.Module):  return sum(p.numel() for p in model.parameters())

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self): self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_acc(logits, labels):
    return sum(logits.argmax(dim=1) == labels) / logits.shape[0]

def create_backup(folders = None, files = None, backup_dir = 'experiments'):
    if folders is None:
        folders = ['.']
    if files is None:
        files = ['.py', '.txt', '.json','.cfg']

    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for file in os.listdir(folder):
            if file.endswith(tuple(files)):
                if folder != '.':
                    src = join(folder, file)
                    dest = join(backup_dir, folder, file)
                else:
                    src = file
                    dest = join(backup_dir, file)
                os.makedirs(os.path.split(dest)[0], exist_ok=True)
                shutil.copy(src, dest)