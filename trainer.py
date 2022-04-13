import torch
import torch.nn as nn
import torch.optim as optim
from models import MARKModel
from typing import Callable, Union, Tuple
from utils import AverageMeter, get_acc
from tqdm import tqdm
import copy
import itertools
from models import KB

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


def update_kb(model, train_dl, val_dl, criterion, task_id, lr = 2e-1, device=torch.device('cuda')):
    train_update_kb(model, train_dl, val_dl, criterion, task_id, lr, device=device)


def train_update_kb(model, train_dl, val_dl, criterion, task_id, lr, device=torch.device('cuda')):
    e_outer = 10
    e_inner = 30
    k = e_outer
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    kb_k = {}
    grads = dict()
    accs_sum = 0.
    optimizer_kb = optim.SGD(list(model.kb.parameters()), lr=lr, weight_decay=1e-2)

    epoch_bar = tqdm(range(e_outer), total=e_outer)
    for epoch in epoch_bar:
        epoch_bar.update()

        # print("....................................")
        # print ("Epoch" + str(epoch))
        for i, (img, label) in enumerate(train_dl):
            if i > k:
                break
            img = img.to(device)
            label = label.to(device)
            model_cp = copy.deepcopy(model)
            optimizer_kb_temp = optim.SGD(list(model_cp.kb.parameters())+list(model_cp.kbcs[task_id].parameters()), lr=lr * 1., weight_decay=1e-3, momentum = 0)

            _, _ = train_batch(model_cp, img, label, optimizer_kb_temp, criterion, task_id, e_inner, device)
            
            grads[i] = get_grads(model.kb, model_cp.kb)
            
            kb_k[i] = copy.deepcopy(model_cp.kb)
            with torch.no_grad():
                model_cp.eval()
                # Note: Original code doesnt use this
                img_val, label_val = next(iter(val_dl))
                loss, acc = train_batch(model_cp, img_val, label_val, None, criterion, task_id, 1, device)
                model_cp.train()
            loss_meter.update(loss, e_inner)
            acc_meter.update(acc, e_inner)
            # Now based on acc, change grads
            accs_sum += acc
            grads[i] = {k:v * acc for k,v in grads[i].items()}
        if accs_sum == 0:
            print('OH NOOOO')
        k_temp = i
        for i in range(k_temp):
            try:
                grads[i] = {dict_k:v / (accs_sum * k_temp) for dict_k,v in grads[i].items()}
            except Exception as e:
                print(e)

        final_grads = dict()
        try:
            final_grads = {n:sum([grads[i][n] for i in range(k_temp)])  for n in grads[0]}
        except Exception as e:
            print(e)
        optimizer_kb.zero_grad()
        for name,param in model.kb.named_parameters():
            if final_grads[name].max() > 100:
                print('Loss going to infinite')
            param.grad = final_grads[name].to(device)
        optimizer_kb.step()
        epoch_bar.set_postfix(loss=loss_meter.avg, accuracy=acc_meter.avg)
        epoch_bar.set_description(f'Epoch: {epoch + 1}')
        epoch_bar.update()


def train_batch(model, img, label, optimizer, criterion, task_id, num_epochs, device = torch.device('cuda')):
    # epoch_bar = tqdm(range(num_epochs), total=num_epochs)
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    img = img.to(device)
    label = label.to(device)
    for epoch in range(num_epochs):
        if optimizer is not None:
            optimizer.zero_grad()
        logits = model(img, task_id)
        loss = criterion(logits, label)
        if optimizer is not None:
            loss.backward()
            for i in range(len(optimizer.param_groups)):
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[i]['params'], 1)
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
            if (loss.item() > 100):
                print("Prob is here")
            loss.backward()
            for i in range(len(optimizer.param_groups)):
                torch.nn.utils.clip_grad_norm_(optimizer.param_groups[i]['params'], 10)

            optimizer.step()

            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(get_acc(logits,label).item(), img.shape[0])
        epoch_bar.set_postfix(loss = loss_meter.avg, accuracy = acc_meter.avg)
        epoch_bar.set_description(f'Epoch: {epoch+1}')
        epoch_bar.update()
from typing import Dict
def get_grads(model_old : KB, model_new : KB) -> Dict[str, torch.nn.Parameter]:
    grads = dict()
    devi = torch.device('cpu')
    for (_, params_old), (name, params_new) in zip(model_old.named_parameters(), model_new.named_parameters()):
        grads[name] = params_new.detach().to(devi) - params_old.detach().to(devi)
    return grads

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