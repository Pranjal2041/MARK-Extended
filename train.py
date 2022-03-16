import torch
import torch.nn as nn
import os

from typing import List

from models import FeatureExtractor, FeatureExtractorClassifier, KBClassifier, KB, MaskGenerator
from models import MARKModel
from trainer import test_loop
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
        # Note: 50 epochs are more than enough
        print(f'Training Feature Extractor on task id {task_id}')
        train_fe(model, train_dl, criterion, task_id, device = device, lr = 1e-1, num_epochs = 20)
        loss, acc = test_loop(lambda x, task_id : model.fecs[task_id](model.fes[task_id](x)), val_dl, task_id, device = device)
        print(f'Feature Extraction Validation Loss {loss} & Accuracy: {acc}')
        
        print('\n---------------------------------------------------------\n')

        # Step 2: Now if task is 0, train the KB without Meta-Learning
        if task_id == 0:
            print(f'Training Initial Knowledge Base Weights')
            train_kb_nonmeta(model, train_dl, criterion, task_id, device = device)
            loss, acc = test_loop(model, val_dl, task_id, device = device)
            print(f'Initial KB Validation Loss {loss} & Accuracy: {acc}')

            print('\n---------------------------------------------------------\n')

        # Step 3: Now train MaskGenerator and Classifier
        print(f'Training Mask Generator and Classifer')
        train_mg_n_clf(model, train_dl, criterion, task_id, device = device)
        loss, acc = test_loop(model, val_dl, task_id, device = device)
        print(f'Mask Generation and Classifer Validation Loss {loss} & Accuracy: {acc}')

        print('\n---------------------------------------------------------\n')

        # TODO: Stage 4: update the KB using meta-learning

    
        # TODO: Stage 5: Fine-Tune Mask Generator and Final Classfier
        print(f'Fine-Tune Mask Generator and Classifer')
        train_mg_n_clf(model, train_dl, criterion, task_id, device = device)
        loss, acc = test_loop(model, val_dl, task_id, device = device)
        print(f'Fine-Tuning Mask Generation and Classifer Validation Loss {loss} & Accuracy: {acc}')


        print('\n---------------------------------------------------------\n')


        # Now report the numbers for task = 0 upto task = task_id
        print(f'Evaluating on tasks {0} to {task_id}\n')
        for t_id in range(task_id+1):
            loss, acc = test_loop(model, test_dl, t_id, device = device)
            print(f'Testing Loss for task = {t_id}: {loss}')
            print(f'Testing Accuracy for task = {t_id}: {acc}\n')

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
   
    train_dl = get_cifar100_dataloader(num_workers=0, batch_size = 128) # Train dl
    test_dl = get_cifar100_dataloader(isTrain=False, num_workers=0, batch_size = 128) # Test dl
    val_dl = get_cifar100_dataloader(isTrain=False, isValid=True, num_workers=0, batch_size = 128) # Valid dl

    train(model, criterion, train_dl, val_dl, test_dl, device)



if __name__ == '__main__':
    main()