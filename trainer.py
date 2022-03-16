import torch
import torch.nn as nn
import torch.optim as optim
from models import MARKModel
from typing import Callable, Union
from utils import AverageMeter, get_acc
from tqdm import tqdm

# TODO: Also Include Validation in these functions itself
def train_mg_n_clf(model : MARKModel, dl, criterion, task_id, lr = 1e-1, num_epochs = 50, device = torch.device('cuda')):
    optimizer = get_optimizer(model, 2, lr = lr, weight_decay = 1e-2, task_id = task_id)
    train_loop(model, dl, optimizer, criterion, task_id, num_epochs, device)

def train_kb_nonmeta(model : MARKModel, dl, criterion, task_id, lr = 1e-1, num_epochs = 50, device = torch.device('cuda')):

    optimizer = get_optimizer(model, 1, lr = lr, weight_decay = 1e-2, task_id = task_id)
    train_loop(model, dl, optimizer, criterion, task_id, num_epochs, device)

def train_fe(model : MARKModel, dl, criterion, task_id, lr = 1e-1, num_epochs=50, device = torch.device('cuda')):
    optimizer = get_optimizer(model, 0, lr = lr, weight_decay = 1e-2, task_id = task_id )
    model_fn = lambda x, task_id: model.fecs[task_id](model.fes[task_id](x))
    train_loop(model_fn, dl, optimizer, criterion, task_id, num_epochs = num_epochs, device = device)


def train_loop(model : Union[MARKModel,Callable[[torch.tensor, int], torch.tensor]], dl, optimizer, criterion, task_id, num_epochs=50, device = torch.device('cuda')):
    epoch_bar = tqdm(range(num_epochs), total = num_epochs)
    for epoch in epoch_bar:
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        for _, (img, label) in enumerate(dl):
            img = img.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            logits = model(img, task_id)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(get_acc(logits,label).item(), img.shape[0])
        epoch_bar.set_postfix(loss = loss_meter.avg, accuracy = acc_meter.avg)
        epoch_bar.set_description(f'Epoch: {epoch+1}')
        epoch_bar.update()




'''
STAGE is one of:
    0 : Initial FE Training
    1 : KB Training for task 0
    2 : Mask n Classifier Training
    3 : KB Update Using Meta-Learning (not-implemented)
'''
def get_optimizer(model : MARKModel, STAGE : int, lr : float = 1e-1, weight_decay : float = 1e-2, task_id : int = 0):
    if STAGE == 0:
        fe = model.fes[task_id]
        fec = model.fecs[task_id]
        optimizer = optim.SGD(list(fe.parameters()) + list(fec.parameters()), lr = lr)
    elif STAGE == 1:
        kb = model.kb
        kbc = model.kbcs[task_id]
        optimizer = optim.AdamW(list(kb.parameters()) + list(kbc.parameters()), lr = lr, weight_decay = weight_decay)
    elif STAGE == 2:
        mg = model.mgs[task_id]
        kbc = model.kbcs[task_id]
        optimizer = optim.AdamW(list(mg.parameters()) + list(kbc.parameters()), lr = lr, weight_decay=weight_decay)
    return optimizer