import torch
import torch.nn as nn
import torch.optim as optim
from models import MARKModel
from typing import Callable, Union, Tuple
from utils import AverageMeter, get_acc
from tqdm import tqdm
import copy
import itertools

# TODO: Also Include Validation in these functions itself

def train_mg_n_clf(model : MARKModel, dl, criterion, task_id, lr = 1e-2, num_epochs = 50, device = torch.device('cuda')):
    optimizer = get_optimizer(model, 2, weight_decay = 1e-2, lr = lr, task_id = task_id)
    train_loop(model, dl, optimizer, criterion, task_id, num_epochs, device)


def train_kb_nonmeta(model : MARKModel, dl, criterion, task_id, lr = 1e-1, num_epochs = 50, device = torch.device('cuda')):
    optimizer = get_optimizer(model, 1, lr = lr, weight_decay = 1e-2, task_id = task_id)
    train_loop(model, dl, optimizer, criterion, task_id, num_epochs, device)


def train_fe(model : MARKModel, dl, criterion, task_id, lr = 1e-1, num_epochs=50, device = torch.device('cuda')):
    optimizer = get_optimizer(model, 0, lr = lr, weight_decay = 1e-2, task_id = task_id )
    model_fn = lambda x, task_id: model.fecs[task_id](model.fes[task_id](x))
    train_loop(model_fn, dl, optimizer, criterion, task_id, num_epochs = num_epochs, device = device)


def update_kb(model, train_dl, val_dl, criterion, task_id, lr = 1e-3, device=torch.device('cuda')):
    optimizer_kb = optim.SGD(list(model.kb.parameters()), lr=lr, weight_decay=1e-2)
    optimizer_kbcs = optim.SGD(list(model.kbcs[task_id].parameters()), lr=lr, weight_decay=1e-2)
    train_update_kb(model, train_dl, val_dl, optimizer_kb, optimizer_kbcs, criterion, task_id, lr, device=device)


def train_update_kb(model, train_dl, val_dl, optimizer_kb, optimizer_kbcs, criterion, task_id, lr, device=torch.device('cuda')):
    e_outer = 15
    e_inner = 5
    k = 10
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    kb_k = {}
    gamma = {}
    kb = copy.deepcopy(model.kb)
    clf = copy.deepcopy(model.kbcs[task_id])
    epoch_bar = tqdm(range(e_outer), total=e_outer)
    for epoch in epoch_bar:
        for i, (img, label) in enumerate(train_dl):
            if i > k:
                break
            img = img.to(device)
            label = label.to(device)
            loss, acc = train_batch(model, img, label, optimizer_kb, criterion, task_id, e_inner, device)
            loss_meter.update(loss, e_inner)
            acc_meter.update(acc, e_inner)
            gamma[i] = get_gamma(model, val_dl, k, i, task_id, criterion)
            kb_k[i] = copy.deepcopy(model.kb)
            model.kb = copy.deepcopy(kb)

            train_batch(model, img, label, optimizer_kbcs, criterion, task_id, e_inner, device)
            model.kbcs[task_id] = copy.deepcopy(clf)

        # update KB

        delta_kb = {}
        delta_kb['diff'] = []
        for i in range(k):
            grads = {}
            for (_, p_new), (n, p_old) in zip(kb_k[i].named_parameters(), model.kb.named_parameters()):
                grads[n] = (p_new - p_old)
            delta_kb['diff'].append(grads)

        delta_kb_final = {}

        for n, p in model.kb.named_parameters():
            delta_kb_final[n] = torch.zeros_like(p)

        for i, g in enumerate(delta_kb['diff']):
            for n in g:
                delta_kb_final[n] += g[n] * (gamma[i]) / k

        for n, p in model.kb.named_parameters():
            if n in delta_kb_final:
                p.grad = delta_kb_final[n]

        optimizer_kb.step()
        optimizer_kb.zero_grad()
        epoch_bar.set_postfix(loss=loss_meter.avg, accuracy=acc_meter.avg)
        epoch_bar.set_description(f'Epoch: {epoch + 1}')
        epoch_bar.update()


def train_batch(model, img, label, optimizer, criterion, task_id, num_epochs, device = torch.device('cuda')):
    # epoch_bar = tqdm(range(num_epochs), total=num_epochs)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for epoch in range(num_epochs):
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        logits = model(img, task_id)
        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), img.shape[0])
        acc_meter.update(get_acc(logits, label).item(), img.shape[0])
        # epoch_bar.set_postfix(loss=loss.item(), accuracy=get_acc(logits,label).item())
        # epoch_bar.set_description(f'Epoch: {epoch + 1}')
        # epoch_bar.update()
    return loss_meter.avg, acc_meter.avg


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


def get_gamma(model, val_dl, j, k, task_id, criterion, device = torch.device('cuda')):
    accs = 0.0
    acc_k = 0.0
    for i, (img, label) in enumerate(val_dl):
    #for i, (img, label) in itertools.islice(val_dl, stop=j):
        if i > j:
            break
        #img = img.to(device)
        #label = label.to(device)
        logits = model(img, task_id)
        loss = criterion(logits, label)
        acc = get_acc(logits, label).item()
        if i == k:
            acc_k = acc
        accs += acc
    return acc_k/accs


def test_loop(model : Union[MARKModel, Callable[[torch.tensor, int], torch.tensor]], dl, task_id, criterion = nn.CrossEntropyLoss(), device = torch.device('cuda')) -> Tuple[float, float]:
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    for _, (img, label) in enumerate(dl):
        img = img.to(device)
        label = label.to(device)

        logits = model(img, task_id)
        loss = criterion(logits, label)

        loss_meter.update(loss.item(), img.shape[0])
        acc_meter.update(get_acc(logits,label).item(), img.shape[0])
    return loss_meter.avg, acc_meter.avg


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
        optimizer = optim.SGD(list(kb.parameters()) + list(kbc.parameters()), lr = lr, weight_decay = weight_decay)
    elif STAGE == 2:
        mg = model.mgs[task_id]
        kbc = model.kbcs[task_id]
        optimizer = optim.SGD(list(mg.parameters()) + list(kbc.parameters()), lr = lr, weight_decay=weight_decay)
    elif STAGE == 3: # Not Required - need to get different optimizer for KB and classifier
        #kb = model.kb
        #kbc = model.kbcs[task_id]
        #optimizer = optim.SGD(list(kb.parameters()), lr = lr, weight_decay=weight_decay)
        raise NotImplementedError('Implement KB Meta-Update')

    return optimizer