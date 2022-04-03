import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
from utils import list_to_device
from advanced_config import AdvancedConfig
import transformers


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
        self.fc_1 = nn.Linear(hidden_dims[-1], output_dim)
        self.fc_2 = nn.Linear(hidden_dims[-1], output_dim)
        self.fc_3 = nn.Linear(hidden_dims[-1], output_dim)
        self.fc_4 = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x, lengths, masks=None):
        x = x.float()
        size = x.size(0)
        for i, lstm in enumerate(self.lstms):
            hidden = torch.zeros(1, size, lstm.hidden_size)
            cell = torch.zeros(1, size, lstm.hidden_size)
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            x, _ = lstm(x, (hidden, cell))
            x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x_1 = self.fc_1(x)
        x_2 = self.fc_2(x)
        x_3 = self.fc_3(x)
        x_4 = self.fc_4(x)
        return x_1, x_2, x_3, x_4


class FeatureExtractorClassifier(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.fc_1 = nn.Linear(input_dim, output_dim)
        self.fc_2 = nn.Linear(input_dim, output_dim)
        self.fc_3 = nn.Linear(input_dim, output_dim)
        self.fc_4 = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc_1(x), self.fc_2(x), self.fc_3(x), self.fc_4(x)


class FeatExtractorLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.linear_1 = nn.Linear(hidden_size, output_size)
        self.linear_2 = nn.Linear(hidden_size, output_size)
        self.linear_3 = nn.Linear(hidden_size, output_size)
        self.linear_4 = nn.Linear(hidden_size, output_size)

    def forward(self, x, lengths):
        if self.input_size == 1: x = x.unsqueeze(-1)
        x = x.float()
        packed_seq = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        hidden = torch.zeros(1, x.size(0), self.hidden_size)
        cell = torch.zeros(1, x.size(0), self.hidden_size)
        x, _ = self.lstm(packed_seq, (hidden, cell))
        x, _ = nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x_1 = self.linear_1(x)
        x_2 = self.linear_2(x)
        x_3 = self.linear_3(x)
        x_4 = self.linear_4(x)

        return x_1, x_2, x_3, x_4


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


class KBClassifier(nn.Module):
    '''
        input_dim: Indicates the output dimension size from the KB Blocks
        num_classes: List of length num_tasks, each indicating number of classes in that task
        num_tasks: Number of tasks
    '''
    def __init__(self, input_dim : int, num_classes : int):

        super().__init__()

        self.num_classes = num_classes
        self.input_dim = input_dim
        self.fc_1 = nn.Linear(input_dim, num_classes)
        self.fc_2 = nn.Linear(input_dim, num_classes)
        self.fc_3 = nn.Linear(input_dim, num_classes)
        self.fc_4 = nn.Linear(input_dim, num_classes)

    def forward(self, X : torch.tensor) -> torch.tensor:
        # TODO: Need to flatten or something for the input?
        return self.fc_1(X), self.fc_2(X), self.fc_3(X), self.fc_4(X),

# TODO: Use Config Instead of PARAMS
# TODO: need to update for LSTM


class MARKModel(nn.Module):

    def __init__(self, cfg: AdvancedConfig, device : torch.device):

        self.FE_HIDDEN = cfg.MODEL.FE_HIDDEN
        self.EMBED_DIM = cfg.MODEL.EMBED_DIM
        self.DIMENSIONS = cfg.MODEL.DIMENSIONS
        self.NUM_CLASSES = cfg.DATASET.NUM_CLASSES
        self.NUM_TASKS = cfg.DATASET.NUM_TASKS
        super().__init__()
        self.fes = list_to_device([FeatExtractorLSTM(3, self.FE_HIDDEN, self.EMBED_DIM) for _ in range(self.NUM_TASKS)], device)
        self.fecs = list_to_device([FeatureExtractorClassifier(self.EMBED_DIM, self.NUM_CLASSES) for _ in range(self.NUM_TASKS)], device)
        self.mgs = list_to_device([MaskGenerator(self.EMBED_DIM, self.DIMENSIONS) for _ in range(self.NUM_TASKS)], device)
        self.kb = LSTMKB((3,32,32), self.EMBED_DIM, self.DIMENSIONS).to(device)
        self.kbcs = list_to_device([KBClassifier(self.EMBED_DIM, 5) for _ in range(self.NUM_TASKS)], device)

    def forward(self, X : torch.tensor, task_id : int) -> torch.tensor:
        # Full model
        X_ = self.fes[task_id](X)
        masks = self.mgs[task_id](X_)
        X_ = self.kb(X, masks)
        logits = self.kbcs[task_id](X_)
        return logits