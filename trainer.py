import torch
import torch.nn as nn
import torch.optim as optim
from models import MARKModel

# TODO: Also Include Validation in these functions itself
def train_mg_n_clf(model : MARKModel, dl, criterion, task_id, lr = 1e-1, num_epochs = 50, device = torch.device('cuda')):

    mg = model.mgs[task_id]
    kbc = model.kbcs[task_id]

    optimizer = optim.AdamW(list(mg.parameters()) + list(kbc.parameters()), lr = lr, weight_decay=1e-2)

    for epoch in range(num_epochs):
        correct = 0
        loss_ = 0.0
        length = 0
        for i, (img, label) in enumerate(dl):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            logits = model(img, task_id)
            pred = torch.max(logits, 1)[1]
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            correct += (pred == label).float().sum().item()
            loss_ += loss.item()
            length += img.shape[0]
        acc = correct / length
        print('Epoch [{}/{}], Loss: {:.7f}, Accuracy: {:.4f}'.format(epoch + 1, num_epochs, loss_ / (i+1), acc))

def train_kb_nonmeta(model : MARKModel, dl, criterion, task_id, lr = 1e-1, num_epochs = 50, device = torch.device('cuda')):

    kb = model.kb
    kbc = model.kbcs[task_id]

    optimizer = optim.AdamW(list(kb.parameters()) + list(kbc.parameters()), lr = lr)

    for epoch in range(num_epochs):
        correct = 0
        loss_ = 0.0
        length = 0
        for i, (img, label) in enumerate(dl):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            logits = model(img, task_id)
            pred = torch.max(logits, 1)[1]
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            correct += (pred == label).float().sum().item()
            loss_ += loss.item()
            length += img.shape[0]
        acc = correct / length
        print('Epoch [{}/{}], Loss: {:.7f}, Accuracy: {:.4f}'.format(epoch + 1, num_epochs, loss_ / (i+1), acc))


def train_fe(model : MARKModel, dl, criterion, task_id, lr = 1e-1, num_epochs=50, device = torch.device('cuda')):
    # TODO: Create AverageMeter and Other Utils

    fe = model.fes[task_id]
    fec = model.fecs[task_id]
    optimizer = optim.SGD(list(fe.parameters()) + list(fec.parameters()), lr = lr)

    for epoch in range(num_epochs):
        correct = 0
        loss_ = 0.0
        length = 0
        for i, (img, label) in enumerate(dl):
            img = img.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            logits = fec(fe(img))
            pred = torch.max(logits, 1)[1]
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            correct += (pred == label).float().sum().item()
            loss_ += loss.item()
            length += img.shape[0]
        acc = correct / length
        print('Epoch [{}/{}], Loss: {:.7f}, Accuracy: {:.4f}'.format(epoch + 1, num_epochs, loss_ / (i+1), acc))
