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


def update_kb(model, train_dl, val_dl, criterion, task_id, lr = 1e-2, device=torch.device('cuda')):

    # optimizer_kb = optim.SGD(list(model.kb.parameters()), lr=lr, weight_decay=1e-2)
    # optimizer_kbcs = optim.SGD(list(model.kbcs[task_id].parameters()), lr=lr, weight_decay=1e-2)
    # print('BeforeF', sum([x.sum() for x in model.kb.parameters()]))
    # print('BeforeF', sum([x.sum() for x in model.kb.parameters()]))
    train_update_kb(model, train_dl, val_dl, criterion, task_id, lr, device=device)
    # print('AfterF', sum([x.sum() for x in model.kb.parameters()]))
    # print('AfterF', sum([x.sum() for x in model.kb.parameters()]))
    # return model


# def train_update_kb(model, train_dl, val_dl, optimizer_kb, criterion, task_id, lr, device=torch.device('cuda')):
#     e_outer = 2
#     e_inner = 10
#     k = 10
#     loss_meter = AverageMeter()
#     acc_meter = AverageMeter()
#     kb_k = {}
#     gamma = {}
#     # optim_kb = copy.deepcopy(optimizer_kb)
#     # optim_clf = copy.deepcopy(optimizer_kbcs)
#     # clf = copy.deepcopy(model.kbcs[task_id])
#     epoch_bar = tqdm(range(e_outer), total=e_outer)
#     for epoch in epoch_bar:
#         kb = copy.deepcopy(model.kb)
#         # print("....................................")
#         # print ("Epoch" + str(epoch))
#         for i, (img, label) in enumerate(train_dl):
#             if i > k:
#                 break
#             img = img.to(device)
#             label = label.to(device)
#             # print ('Before', sum([x.sum() for x in model.parameters()]))
#             # print('Before', sum([x.sum() for x in model.kb.parameters()]))
#             # loss, acc = train_batch(model, img, label, optimizer_kb, criterion, task_id, e_inner, device)
#             optimizer_kb_temp = optim.SGD(list(model.kb.parameters()), lr=lr, weight_decay=1e-2)
#             _, _ = train_batch(model, img, label, optimizer_kb_temp, criterion, task_id, e_inner, device)
#             # print('After', sum([x.sum() for x in model.parameters()]))
#             # print('After', sum([x.sum() for x in model.kb.parameters()]))
#             # loss_meter.update(loss, e_inner)
#             # acc_meter.update(acc, e_inner)
#             # print("Loss" + str(loss))
#             # print("acc" + str(acc))
#
#             gamma[i] = get_gamma(model, val_dl, k, i, task_id, criterion, device)
#             # x=list(model.kb.parameters())
#             kb_k[i] = copy.deepcopy(model.kb)
#             model.kb = copy.deepcopy(kb)
#             # x1=list(kb_k[i].parameters())
#             # x2=list(model.kb.parameters())
#             # kbt = model.kb
#             # optimizer_kb = optim.SGD(list(model.kb.parameters()), lr=lr, weight_decay=1e-2)
#
#
#             loss, acc = train_batch(model, img, label, None, criterion, task_id, e_inner, device)
#             loss_meter.update(loss, e_inner)
#             acc_meter.update(acc, e_inner)
#             # model.kbcs[task_id] = copy.deepcopy(clf)
#             # optimizer_kbcs = copy.deepcopy(optim_clf)
#
#         # update KB
#         delta_kb = {}
#         delta_kb['diff'] = []
#
#         for i in range(k):
#             grads = {}
#             for (_, p_new), (n, p_old) in zip(kb_k[i].named_parameters(), model.kb.named_parameters()):
#                 grads[n] = (p_new - p_old)
#             delta_kb['diff'].append(grads)
#
#         delta_kb_final = {}
#
#         for n, p in model.kb.named_parameters():
#             delta_kb_final[n] = torch.zeros_like(p)
#
#         for i, g in enumerate(delta_kb['diff']):
#             for n in g:
#                 delta_kb_final[n] += g[n] * (gamma[i]) / k
#
#         for n, p in model.kb.named_parameters():
#             if n in delta_kb_final:
#                 p.grad = delta_kb_final[n]
#
#         # model.train()
#         optimizer_kb = optim.SGD(list(model.kb.parameters()), lr=lr, weight_decay=1e-2)
#
#         optimizer_kb.step()
#         optimizer_kb.zero_grad()
#
#
#         epoch_bar.set_postfix(loss=loss_meter.avg, accuracy=acc_meter.avg)
#         epoch_bar.set_description(f'Epoch: {epoch + 1}')
#         epoch_bar.update()
#     # return model


def train_update_kb(model, train_dl, val_dl, criterion, task_id, lr, device=torch.device('cuda')):
    e_outer = 15
    e_inner = 40
    k = 10
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    kb_k = {}
    gamma = {}
    # optim_kb = copy.deepcopy(optimizer_kb)
    # optim_clf = copy.deepcopy(optimizer_kbcs)
    # clf = copy.deepcopy(model.kbcs[task_id])
    epoch_bar = tqdm(range(e_outer), total=e_outer)
    for epoch in epoch_bar:

        # print("....................................")
        # print ("Epoch" + str(epoch))
        for i, (img, label) in enumerate(train_dl):
            if i > k:
                break
            img = img.to(device)
            label = label.to(device)
            kb = copy.deepcopy(model)
            # print ('Before', sum([x.sum() for x in model.parameters()]))
            # print('Before', sum([x.sum() for x in model.kb.parameters()]))
            # loss, acc = train_batch(model, img, label, optimizer_kb, criterion, task_id, e_inner, device)
            # l=list(kb.parameters())
            # l1=sum([x.sum() for x in kb.kb.parameters()])
            optimizer_kb_temp = optim.SGD(list(kb.kb.parameters()), lr=lr, weight_decay=1e-2, momentum = 0)

            _, _ = train_batch(kb, img, label, optimizer_kb_temp, criterion, task_id, e_inner, device)
            # l2 = sum([x.sum() for x in kb.kb.parameters()])
            # x=list(kb.kb.parameters())
            # y=list(kb.kb.named_parameters())
            # print('After', sum([x.sum() for x in model.parameters()]))
            # print('After', sum([x.sum() for x in model.kb.parameters()]))
            # loss_meter.update(loss, e_inner)
            # acc_meter.update(acc, e_inner)
            # print("Loss" + str(loss))
            # print("acc" + str(acc))

            gamma[i] = get_gamma(kb, val_dl, k, i, task_id, criterion, device)
            # x=list(model.kb.parameters())
            kb_k[i] = copy.deepcopy(kb.kb)
            # model.kb = copy.deepcopy(kb)
            # x1=list(kb_k[i].parameters())
            # x2=list(model.kb.parameters())
            # kbt = model.kb
            # optimizer_kb = optim.SGD(list(model.kb.parameters()), lr=lr, weight_decay=1e-2)
            loss, acc = train_batch(kb, img, label, None, criterion, task_id, e_inner, device)
            loss_meter.update(loss, e_inner)
            acc_meter.update(acc, e_inner)
            # model.kbcs[task_id] = copy.deepcopy(clf)
            # optimizer_kbcs = copy.deepcopy(optim_clf)

        # update KB
        delta_kb = {}
        delta_kb['diff'] = []
        optimizer_kb = optim.SGD(list(model.kb.parameters()), lr=lr, weight_decay=1e-2)
        for i in range(k):
            grads = {}
            kl = []
            for (_, p_new), (n, p_old) in zip(kb_k[i].named_parameters(), model.kb.named_parameters()):
                grads[n] = (p_new - p_old)
                kl.append(sum([x.sum() for x in (p_new - p_old)]))
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
    # return model


def train_batch(model, img, label, optimizer, criterion, task_id, num_epochs, device = torch.device('cuda')):
    # epoch_bar = tqdm(range(num_epochs), total=num_epochs)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for epoch in range(num_epochs):
        img = img.to(device)
        label = label.to(device)
        if optimizer is not None:
            optimizer.zero_grad()
        logits = model(img, task_id)
        loss = criterion(logits, label)
        if optimizer is not None:
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
        img = img.to(device)
        label = label.to(device)
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
    dl.dataset.set_task(task_id)
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