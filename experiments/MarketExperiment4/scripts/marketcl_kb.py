from dataloaders import get_marketcl_dataloader
import torch
from torch import nn
from torch import optim
from typing import List


class LSTMKB(nn.Module):

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        self.lstms: nn.ModuleList = nn.ModuleList()
        self.lstms.append(
            nn.LSTM(input_dim, hidden_dims[0], batch_first=True))
        for h in hidden_dims[1:]:
            self.lstms.append(
                nn.LSTM(self.lstms[-1].hidden_size, h, batch_first=True))    

    def forward(self, x, lengths, masks=None):
        x = x.float()
        for i, lstm in enumerate(self.lstms):
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            x, _ = lstm(x, )
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

            if masks is not None:
                x = x*masks[i].unsqueeze(0).unsqueeze(0)
        x_final = x[:, -1, :]
        return x_final

class KBMarketClassifier(nn.Module):

    def __init__(self, input_dim, num_classes = 4, num_labels = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.fcs = nn.ModuleList([nn.Linear(input_dim, num_classes) for _ in range(self.num_labels)])

    def forward(self, X):
        outs = []
        for i in range(self.num_labels):
            outs.append(self.fcs[i](X))
        return outs

def Train(net, classifier, market_dl, criterion=nn.CrossEntropyLoss(), epochs=5):
    optimizer = optim.Adam(list(net.parameters()) + list(classifier.parameters()), lr=1e-2)

    for epoch in range(epochs):
        correct_1 = correct_2 = correct_3 = correct_4 = 0
        loss_ = 0.0
        length = 0
        # label_ratios = {0:0, 1:0, 2:0, 3:0}
        for i, (feat, label, lengths) in enumerate(market_dl):
            # for l in label[:,0]:
            #     label_ratios[int(l.item())] += 1 
            feat = feat.cuda()
            label = label.cuda()
            optimizer.zero_grad()
            logits_arr = classifier(net(feat, lengths))
            loss = sum([criterion(logits_arr[i], label[:,i]) for i in range(4)])
            loss.backward()
            optimizer.step()
            #if i == 5:
            #    break
            correct_1 += (torch.max(logits_arr[0], 1)[1] == label[:, 0]).float().sum().item()
            correct_2 += (torch.max(logits_arr[1], 1)[1] == label[:, 1]).float().sum().item()
            correct_3 += (torch.max(logits_arr[2], 1)[1] == label[:, 2]).float().sum().item()
            correct_4 += (torch.max(logits_arr[3], 1)[1] == label[:, 3]).float().sum().item()
            loss_ += loss.item()
            length += feat.shape[0]
        acc_1 = correct_1 / length
        acc_2 = correct_2 / length
        acc_3 = correct_3 / length
        acc_4 = correct_4 / length
        print('Epoch [{}/{}], Loss: {:.7f}, Accuracy 1: {:.4f}, Accuracy 2: {:.4f}, Accuracy 3: {:.4f}, Accuracy 4: {:.4f}'.format(epoch + 1, epochs, loss_ / length, acc_1, acc_2, acc_3, acc_4))
        # print(label_ratios)
    return net


if __name__ == '__main__':

    market_dl = get_marketcl_dataloader()
    NUM_TASKS = 1
    NUM_DAYS = 5
    fe = {}
    for task_id in range(NUM_TASKS):
        market_dl.dataset.set_symbol(task_id)
        model = LSTMKB(input_dim=28, hidden_dims=[64, 128, 256], output_dim=4)
        classifier = KBMarketClassifier(256)
        model.cuda()
        classifier.cuda()
        for day in range(NUM_DAYS):
            print(f"Training KB for task id: {task_id}, day {day}")
            market_dl.dataset.set_day(day)
            model = Train(model, classifier, market_dl)
        fe[task_id] = model