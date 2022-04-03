from dataloaders import get_marketcl_dataloader
import torch
from torch import nn
from torch import optim


class FeatExtractorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, lr):
        super().__init__()
        ### INSERT YOUR CODE HERE
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear_1 = nn.Linear(hidden_size, output_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.linear_3 = nn.Linear(hidden_size, output_size)
        self.linear_4 = nn.Linear(hidden_size, output_size)
        self.logsoft = nn.LogSoftmax(dim=-1)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        if self.input_size == 1: x = x.unsqueeze(-1)
        x = x.float()
        hidden = torch.zeros(1, x.size(0), self.hidden_size)
        cell = torch.zeros(1, x.size(0), self.hidden_size)
        x, _ = self.lstm(x, (hidden, cell))
        x_1 = self.linear_1(x)
        x_1 = self.logsoft(x_1)
        x_1 = x_1[:, -1, :]

        x_2 = self.linear_2(x)
        x_2 = self.logsoft(x_2)
        x_2 = x_2[:, -1, :]

        x_3 = self.linear_3(x)
        x_3 = self.logsoft(x_3)
        x_3 = x_3[:, -1, :]

        x_4 = self.linear_4(x)
        x_4 = self.logsoft(x_4)
        x_4 = x_4[:, -1, :]

        return x, x_1, x_2, x_3, x_4


def TrainFeature(net, market_dl, Loss=nn.NLLLoss(), epochs=5):
    optimizer = net.optimizer
    for epoch in range(epochs):
        correct_1 = correct_2 = correct_3 = correct_4 = 0
        loss_ = 0.0
        length = 0
        for i, (feat, label, lengths) in enumerate(market_dl):
            # print(feat.shape)
            # print(label.shape)
            # print(lengths.shape)
            optimizer.zero_grad()
            _, predictions_1, predictions_2, predictions_3, predictions_4 = net(feat)
            pred_1 = torch.max(predictions_1, 1)[1]
            pred_2 = torch.max(predictions_2, 1)[1]
            pred_3 = torch.max(predictions_3, 1)[1]
            pred_4 = torch.max(predictions_4, 1)[1]
            loss = Loss(predictions_1, label[:, 0]) + Loss(predictions_2, label[:, 1]) + Loss(predictions_3, label[:, 2]) + Loss(predictions_4, label[:, 3])
            loss.backward()
            optimizer.step()
            #if i == 5:
            #    break
            correct_1 += (pred_1 == label[:, 0]).float().sum().item()
            correct_2 += (pred_2 == label[:, 1]).float().sum().item()
            correct_3 += (pred_3 == label[:, 2]).float().sum().item()
            correct_4 += (pred_4 == label[:, 3]).float().sum().item()
            loss_ += loss.item()
            length += feat.shape[0]
        acc_1 = correct_1 / length
        acc_2 = correct_2 / length
        acc_3 = correct_3 / length
        acc_4 = correct_4 / length
        print('Epoch [{}/{}], Loss: {:.7f}, Accuracy 1: {:.4f}, Accuracy 2: {:.4f}, Accuracy 3: {:.4f}, Accuracy 4: {:.4f}'.format(epoch + 1, epochs, loss_ / length, acc_1, acc_2, acc_3, acc_4))
    return net


if __name__ == '__main__':

    market_dl = get_marketcl_dataloader()
    NUM_TASKS = 5
    NUM_DAYS = 5
    fe = {}
    for task_id in range(NUM_TASKS):
        market_dl.dataset.set_symbol(task_id)
        model = FeatExtractorLSTM(input_size=28, hidden_size=64, output_size=4, lr=1e-3)
        for day in range(NUM_DAYS):
            print(f"training for task id: {task_id}, day {day}")
            market_dl.dataset.set_day(day)
            model = TrainFeature(model, market_dl)
        fe[task_id] = model
        break

