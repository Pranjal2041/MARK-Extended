import torch
from torch import nn
import torch.nn.functional as F
from torch import optim


class FeatureExtractor(nn.Module):
    def __init__(self, sample_dim, input_size, hidden_size, output_size, lr):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.sample_dim = sample_dim
        self.conv_1 = torch.nn.Conv2d(in_channels=input_size, out_channels=hidden_size, kernel_size=(4, 4))
        self.batch_norm = torch.nn.BatchNorm2d(self.hidden_size)
        self.conv_2 = torch.nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(3, 3))
        self.linear = torch.nn.Linear(1152, output_size)
        self.optimizer = optim.SGD(self.parameters(), lr=lr)
        self.logsoft = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = self.conv_2(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        x = F.max_pool2d(x, (2, 2))
        x = x.flatten(start_dim=1, end_dim=-1)
        x = self.linear(x)
        emb = F.relu(x)
        soft = self.logsoft(emb)
        return emb, soft


def TrainFeature(net, data, Loss=nn.NLLLoss(), epochs=50):
    optimizer = net.optimizer
    for epoch in range(epochs):
        correct = 0
        loss_ = 0.0
        length = 0
        for i, (img, label, img_feats) in enumerate(data):
            optimizer.zero_grad()
            emb, predictions = net(img)
            pred = torch.max(predictions, 1)[1]
            loss = Loss(predictions, label)
            loss.backward()
            optimizer.step()
            #if i == 5:
            #    break
            correct += (pred == label).float().sum().item()
            loss_ += loss.item()
            length += img.shape[0]
        acc = correct / length
    print('Epoch [{}/{}], Loss: {:.7f}, Accuracy: {:.4f}'.format(epoch + 1, epochs, loss_ / length, acc))
    return net


# def TrainFeature(net, data, Loss=nn.NLLLoss(), epochs=50):
#     #nets = {}
#     for i, (img, label, img_feats) in enumerate(data):
#         #net_t = net
#         #print(f"Task ID - {i}")
#         optimizer = net.optimizer
#         for epoch in range(epochs):
#             optimizer.zero_grad()
#             emb, predictions = net(img)
#             pred = torch.max(predictions, 1)[1]
#             loss = Loss(predictions, label)
#             loss.backward()
#             optimizer.step()
#             correct = (pred == label).float().sum().item()
#             loss_ = loss.item()
#             length = img.shape[0]
#             acc = correct / length
#         print('Epoch [{}/{}], Loss: {:.7f}, Accuracy: {:.4f}'.format(epochs, epochs, loss_ / length, acc))
#         #nets[i] = net_t
#     return net


def get_feature_embedding(net, data):
    embedding_batches = []
    for i, (img, label, img_feats) in enumerate(data):
        embedding, _ = net(img)
        embedding_batches.append(embedding)
    return embedding_batches


