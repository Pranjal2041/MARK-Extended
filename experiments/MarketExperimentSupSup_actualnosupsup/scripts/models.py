import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple
from utils import list_to_device
from advanced_config import AdvancedConfig
import transformers
from models_supsup import MultitaskMaskLinear
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
    def __init__(self, in_channels, hidden_size, output_size, img_size = 32):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.conv_1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=hidden_size, kernel_size=(4, 4))
        self.batch_norm = torch.nn.BatchNorm2d(self.hidden_size)
        # TODO: Can we change the size of second hidden_dim
        self.conv_2 = torch.nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(3, 3))
        
        # TODO: Make 1152 flexible and take as input
        if img_size == 32:
            self.linear = torch.nn.Linear(1152, output_size)
        else:
            self.linear = torch.nn.Linear(288, output_size)

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
    def __init__(self, feat_dim : int, dimensions : List[int], supsup = False, num_tasks = -1,):

        super().__init__()

        self.feat_dim = feat_dim
        self.dimensions = dimensions
        self.num_masks = len(dimensions)
        self.supsup = supsup
        self.num_tasks = num_tasks
        fcs : nn.ModuleList = nn.ModuleList()
        
        

        for i in range(self.num_masks):
            fcs.append(nn.Sequential(
                nn.Linear(self.feat_dim, dimensions[i]) if not self.supsup else MultitaskMaskLinear(self.feat_dim, dimensions[i], num_tasks = self.num_tasks, bias = False),
                nn.ReLU()
            ))
        self.fcs = fcs
    
    def set_task_id(self, task_id):
        for n, m in self.named_modules():
            if isinstance(m, MultitaskMaskLinear):
                m.task = task_id

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
    def __init__(self, input_dim : int, num_classes : int, supsup = False, num_tasks = -1):

        super().__init__()

        self.num_classes = num_classes
        self.input_dim = input_dim
        self.num_tasks = num_tasks
        self.supsup = supsup
        if self.supsup:
            self.fc = MultitaskMaskLinear(input_dim, num_classes, num_tasks = num_tasks, bias = False)
        else:
            self.fc = nn.Linear(input_dim, num_classes)

    def set_task_id(self, task_id):
        for n, m in self.named_modules():
            if isinstance(m, MultitaskMaskLinear):
                m.task = task_id

    def forward(self, X : torch.tensor) -> torch.tensor:
        return self.fc(X)

class MaskedModelsList(list):
    def __getitem__(self, key):
        l = list.__getitem__(self, key)
        l.set_task_id(key)
        return l


# TODO: Use Config Instead of PARAMS
class MARKModel(nn.Module):

    def __init__(self, cfg: AdvancedConfig, device : torch.device, image_size = 32,):

        self.FE_HIDDEN = cfg.MODEL.FE_HIDDEN
        self.EMBED_DIM = cfg.MODEL.EMBED_DIM
        self.DIMENSIONS = cfg.MODEL.DIMENSIONS
        self.NUM_CLASSES = cfg.DATASET.NUM_CLASSES
        self.NUM_TASKS = cfg.DATASET.NUM_TASKS
        self.supsup = cfg.MODEL.SUPSUP
        super().__init__()


        self.fes = list_to_device([FeatureExtractor(3, self.FE_HIDDEN, self.EMBED_DIM, img_size = image_size) for _ in range(self.NUM_TASKS)], device)
        self.kb = KB((3,image_size,image_size), self.EMBED_DIM, self.DIMENSIONS).to(device)
        self.fecs = list_to_device([FeatureExtractorClassifier(self.EMBED_DIM, self.NUM_CLASSES) for _ in range(self.NUM_TASKS)], device)

        if self.supsup:
            # Create 
            mg = MaskGenerator(self.EMBED_DIM, self.DIMENSIONS, supsup = True, num_tasks = self.NUM_TASKS)
            self.mgs = MaskedModelsList(list_to_device([mg for _ in range(self.NUM_TASKS)], device))
            kbc = KBClassifier(self.EMBED_DIM, 5, supsup = True, num_tasks = self.NUM_TASKS)
            self.kbcs = MaskedModelsList(list_to_device([kbc for _ in range(self.NUM_TASKS)], device))
        else:
            self.mgs = list_to_device([MaskGenerator(self.EMBED_DIM, self.DIMENSIONS) for _ in range(self.NUM_TASKS)], device)
            self.kbcs = list_to_device([KBClassifier(self.EMBED_DIM, 5) for _ in range(self.NUM_TASKS)], device)

    def cache_masks(self):
        if not self.supsup:
            return
        for model in [self.mgs[0],self.kbcs[0]]:
            for n, m in model.named_modules():
                if isinstance(m, MultitaskMaskLinear):
                    print(f"=> Caching mask state for {n}")
                    m.cache_masks()

    def forward(self, X : torch.tensor, task_id : int) -> torch.tensor:
        # Full model
        X_ = self.fes[task_id](X)
        masks = self.mgs[task_id](X_)
        X_ = self.kb(X, masks)
        logits = self.kbcs[task_id](X_)
        return logits