import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
from utils import list_to_device
from advanced_config import AdvancedConfig


class LSTMKB(nn.Module):

    def __init__(self, input_dim: int, hidden_dims: List[int]):
        super().__init__()
        self.input_dim = input_dim
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


class FeatureExtractorLSTMClassifier(nn.Module):

    def __init__(self, input_dim, output_dim, num_labels = 4):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_labels = num_labels
        self.fcs = nn.ModuleList([nn.Linear(input_dim, output_dim) for _ in range(self.num_labels)])


    def forward(self, x):
        return [fc(x) for fc in self.fcs]


class FeatExtractorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        ### INSERT YOUR CODE HERE
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x, lengths):
        if self.input_size == 1: x = x.unsqueeze(-1)
        x = x.float()
        packed_seq = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(packed_seq,)
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x_final = x[:,-1,:]
        return x_final


class MaskGenerator(nn.Module):
    '''
        Takes the features and outputs corresponding masks
    '''
    def __init__(self, feat_dim : int, dimensions : List[int]):

        super().__init__()

        self.feat_dim = feat_dim
        self.dimensions = dimensions
        self.num_masks = len(dimensions)

        fcs : nn.ModuleList = nn.ModuleList()
        for i in range(self.num_masks):
            fcs.append(nn.Sequential(
                nn.Linear(self.feat_dim, dimensions[i])#,
                #nn.ReLU()
            ))
        self.fcs = fcs

    def forward(self, feats: torch.tensor):

        masks = []
        for i in range(self.num_masks):
            # TODO: Do we need to flatten the features?
            masks.append(self.fcs[i](feats))

        return masks


class KBLSTMClassifier(nn.Module):

    def __init__(self, input_dim, num_classes = 4, num_labels = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_labels = num_labels
        self.fcs = nn.ModuleList([nn.Linear(input_dim, num_classes) for _ in range(self.num_labels)])

    def forward(self, X):
        return [fc(X) for fc in self.fcs]


# TODO: Use Config Instead of PARAMS
# TODO: need to update for LSTM


class MARKLSTMModel(nn.Module):

    def __init__(self, cfg: AdvancedConfig, device : torch.device):

        super().__init__()

        self.INPUT_FEAT_SIZE = cfg.MODEL.INPUT_FEAT_SIZE
        self.FE_HIDDEN = cfg.MODEL.FE_HIDDEN
        self.DIMENSIONS = cfg.MODEL.DIMENSIONS
        self.NUM_CLASSES = cfg.DATASET.NUM_CLASSES
        self.NUM_TASKS = cfg.DATASET.NUM_TASKS


        self.fes = list_to_device([FeatExtractorLSTM(self.INPUT_FEAT_SIZE, self.FE_HIDDEN) for _ in range(self.NUM_TASKS)], device)
        self.fecs = list_to_device([FeatureExtractorLSTMClassifier(self.FE_HIDDEN, self.NUM_CLASSES) for _ in range(self.NUM_TASKS)], device)
        self.mgs = list_to_device([MaskGenerator(self.FE_HIDDEN, self.DIMENSIONS) for _ in range(self.NUM_TASKS)], device)
        self.kb = LSTMKB(28, self.DIMENSIONS).to(device)
        self.kbcs = list_to_device([KBLSTMClassifier(self.DIMENSIONS[-1], 4) for _ in range(self.NUM_TASKS)], device)

    def loss(self, criterion, logits, labels):
        return sum([criterion(logits[i], labels[:,i]) for i in range(labels.shape[1])])

    def accuracy(self, logits, labels):
        return {i:(torch.max(logits[i], 1)[1] == labels[:, i]).float().sum().item() for i in range(labels.shape[1])}

    # task_id corresponds to the symbol id
    def forward(self, X : torch.tensor, task_id : int) -> torch.tensor:
        # Full model
        X_ = self.fes[task_id](X)
        masks = self.mgs[task_id](X_)
        X_ = self.kb(X, masks)
        logits = self.kbcs[task_id](X_)
        return logits