import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
from utils import list_to_device
from advanced_config import AdvancedConfig
import transformers
class KB(nn.Module):

    # input_dim is of form (n_ch, h, w)
    def __init__(self, input_dim : int, output_dim : int, hidden_dims : List[int]):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        self.convs : nn.ModuleList = nn.ModuleList()

        # TODO: In code, kernel size(and stride) is also flexible, but nothing of this sort was mentioned in the paper
        self.convs.append(nn.Conv2d(in_channels=input_dim[0], out_channels=hidden_dims[0], kernel_size=3, stride=1, padding=1))
        for h in hidden_dims[1:]:
            self.convs.append(nn.Conv2d(in_channels=self.convs[-1].out_channels, out_channels=h, kernel_size=3, stride=1, padding=1))
        
        self.pooling = nn.MaxPool2d(2)
        self.activation = nn.ReLU()

        self.fc = nn.Linear((input_dim[1]//(2**len(hidden_dims)))**2 * hidden_dims[-1]  , output_dim)

    def forward(self, x, masks = None):
        # masks is a list/tensor of individual masks of hidden dimension size
        # Note: Last two dimensions can be one as in paper, or equal to hidden size

        for i, conv in enumerate(self.convs):
            x = self.pooling(self.activation(conv(x)))
            x = x*masks[i].unsqueeze(2).unsqueeze(3) if masks is not None else x
        x = x.reshape(x.shape[0], -1)
        return self.activation(self.fc(x))

class FeatureExtractorClassifier(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()

        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.fc(x)

class FeatureExtractor(nn.Module):

    # Prefer modules over functional activations and pooling for more flexibility?
    def __init__(self, in_channels, hidden_size, output_size):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.conv_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=(4, 4))
        self.batch_norm = torch.nn.BatchNorm2d(self.hidden_size)
        # TODO: Can we change the size of second hidden_dim
        self.conv_2 = torch.nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(3, 3))
        
        # TODO: Make 1152 flexible and take as input
        self.linear = torch.nn.Linear(1152, output_size)

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
        emb = F.relu(x) # Generally No ReLU is used in the last layer!!!
        return emb


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
                nn.Linear(self.feat_dim, dimensions[i]),
                nn.ReLU()
            ))
        self.fcs = fcs
    
    def forward(self, feats : torch.tensor):

        masks  = []
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
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, X : torch.tensor) -> torch.tensor:
        # TODO: Need to flatten or something for the input?
        return self.fc(X)

# TODO: Use Config Instead of PARAMS
class MARKModel(nn.Module):

    def __init__(self, cfg: AdvancedConfig, device : torch.device):

        self.FE_HIDDEN = cfg.MODEL.FE_HIDDEN
        self.EMBED_DIM = cfg.MODEL.EMBED_DIM
        self.DIMENSIONS = cfg.MODEL.DIMENSIONS
        self.NUM_CLASSES = cfg.DATASET.NUM_CLASSES
        self.NUM_TASKS = cfg.DATASET.NUM_TASKS
        super().__init__()
        self.fes = list_to_device([FeatureExtractor(3, self.FE_HIDDEN, self.EMBED_DIM) for _ in range(self.NUM_TASKS)], device)
        self.fecs = list_to_device([FeatureExtractorClassifier(self.EMBED_DIM, self.NUM_CLASSES) for _ in range(self.NUM_TASKS)], device)
        self.mgs = list_to_device([MaskGenerator(self.EMBED_DIM, self.DIMENSIONS) for _ in range(self.NUM_TASKS)], device)
        self.kb = KB((3,32,32), self.EMBED_DIM, self.DIMENSIONS).to(device)
        self.kbcs = list_to_device([KBClassifier(self.EMBED_DIM, 5) for _ in range(self.NUM_TASKS)], device)

    def forward(self, X : torch.tensor, task_id : int) -> torch.tensor:
        # Full model
        X_ = self.fes[task_id](X)
        masks = self.mgs[task_id](X_)
        X_ = self.kb(X, masks)
        logits = self.kbcs[task_id](X_)
        return logits