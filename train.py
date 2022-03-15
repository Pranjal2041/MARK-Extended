import torch
import torch.nn as nn
import os

from typing import List

from models import FeatureExtractor, FeatureExtractorClassifier, KBClassifier, KB, MaskGenerator
from models import MARKModel

from trainer import train_fe, train_kb_nonmeta, train_mg_n_clf
from dataloaders import get_cifar100_dataloader

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(model : MARKModel, criterion, train_dl, val_dl, test_dl, device, num_tasks : int = 20):

    # There are four steps:
    # 1. Train the feature extractor on given task.
    # 2. Train KB if task = 0
    # 3. Train MaskGenerator and Classifier
    # 4. Use Meta-Learning to train the KB, and KBClassifier

    for task_id in range(num_tasks):
        train_dl.dataset.set_task(task_id)
        val_dl.dataset.set_task(task_id)
        test_dl.dataset.set_task(task_id)

        # Step 1: First train the feature extractor on task 
        train_fe(model, train_dl, criterion, task_id, device = device)

        # Step 2: Now if task is 0, train the KB without Meta-Learning
        if task_id == 0:
            train_kb_nonmeta(model, train_dl, criterion, task_id, device = device)

        # Step 3: Now train MaskGenerator and Classifier
        train_mg_n_clf(model, train_dl, criterion, task_id, device = device)


        # Now finally update the KB using meta-learning

    

def main(cfg = None):
    # TODO: Add config loading code
    FE_HIDDEN = 32
    EMBED_DIM = 128
    DIMENSIONS = [64,128,256]
    NUM_CLASSES = 5
    NUM_TASKS = 20

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MARKModel(FE_HIDDEN, EMBED_DIM, DIMENSIONS, NUM_CLASSES, NUM_TASKS, device)
    criterion = nn.CrossEntropyLoss().to(device)
    # Different optimizers will be created as and when needed.
   
    train_dl = get_cifar100_dataloader(num_workers=0) # Train dl
    test_dl = get_cifar100_dataloader(isTrain=False, num_workers=0) # Test dl
    val_dl = get_cifar100_dataloader(isTrain=False, isValid=True, num_workers=0) # Valid dl

    train(model, criterion, train_dl, val_dl, test_dl, device)



if __name__ == '__main__':
    main()