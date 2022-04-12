import torch
import torch.nn as nn
import os
import numpy as np
from typing import List

from models import FeatureExtractor, FeatureExtractorClassifier, KBClassifier, KB, MaskGenerator
from models import MARKModel
from trainer import test_loop
from trainer import train_fe, train_kb_nonmeta, train_mg_n_clf, update_kb
from dataloaders import get_cifar100_dataloader
from advanced_logger import LogPriority

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(model : MARKModel, criterion, train_dl, val_dl, test_dl, experimenter, device, baseline):


    print = experimenter.logger.log

    # There are four steps:
    # 1. Train the feature extractor on given task.
    # 2. Train KB if task = 0
    # 3. Train MaskGenerator and Classifier
    # 4. Use Meta-Learning to train the KB, and KBClassifier 
    SUPSUP = experimenter.cfg.MODEL.SUPSUP
    NUM_TASKS = experimenter.cfg.DATASET.NUM_TASKS
    task_accs = np.zeros((NUM_TASKS, NUM_TASKS))       


    for task_id in range(NUM_TASKS):
        experimenter.start_epoch()

        train_dl.dataset.set_task(task_id)
        val_dl.dataset.set_task(task_id)
        test_dl.dataset.set_task(task_id)

        
        # Step 1: First train the feature extractor on task 
        # Note: 50 epochs are more than enough
        print(f'Training Feature Extractor on task id {task_id}')
        train_fe(model, train_dl, criterion, task_id, device = device, lr = float(experimenter.config.TRAINING.LR.FE), num_epochs = experimenter.config.TRAINING.EPOCHS.FE)
        loss, acc = test_loop(lambda x, task_id : model.fecs[task_id](model.fes[task_id](x)), val_dl, task_id, device = device)
        print(f'Feature Extraction Validation Loss {loss} & Accuracy: {acc}')
        
        print('\n---------------------------------------------------------\n')

        # Step 2: Now if task is 0, train the KB without Meta-Learning
        if task_id == 0 or baseline==1 or baseline==3:
            print(f'Training Initial Knowledge Base Weights')
            train_kb_nonmeta(model, train_dl, criterion, task_id, lr = float(experimenter.config.TRAINING.LR.INIT_KB), device = device, num_epochs = experimenter.config.TRAINING.EPOCHS.INIT_KB)
            loss, acc = test_loop(model, val_dl, task_id, device = device)
            print(f'Initial KB Validation Loss {loss} & Accuracy: {acc}')

            print('\n---------------------------------------------------------\n')

        if baseline==0 or baseline==3:
            # Step 3: Now train MaskGenerator and Classifier
            print(f'Training Mask Generator and Classifier')
            train_mg_n_clf(model, train_dl, criterion, task_id, lr = float(experimenter.config.TRAINING.LR.MG_N_C), device = device, num_epochs = experimenter.config.TRAINING.EPOCHS.MG_N_C)
            loss, acc = test_loop(model, val_dl, task_id, device = device)
            print(f'Mask Generation and Classifier Validation Loss {loss} & Accuracy: {acc}')

            print('\n---------------------------------------------------------\n')
        if baseline == 0 or baseline ==2:
            # '''
            print('Updating KB')
            print('Before',sum([x.sum() for x in model.kb.parameters()]))
            update_kb(model, train_dl, val_dl, criterion, task_id, device=device)
            loss, acc = test_loop(model, val_dl, task_id, device=device)
            print(f'Update KB {loss} & Accuracy: {acc}')
            print('After',sum([x.sum() for x in model.kb.parameters()]))

            print('\n---------------------------------------------------------\n')
            # '''
        if baseline == 0 or baseline ==3:
            # Stage 5: Fine-Tune Mask Generator and Final Classifier
            print(f'Fine-Tune Mask Generator and Classifier')
            train_mg_n_clf(model, train_dl, criterion, task_id, lr = float(experimenter.config.TRAINING.LR.FINETUNE_MG_N_C), num_epochs = experimenter.config.TRAINING.EPOCHS.FINETUNE_MG_N_C, device = device)
            loss, acc = test_loop(model, val_dl, task_id, device = device)
            print(f'Fine-Tuning Mask Generation and Classifier Validation Loss {loss} & Accuracy: {acc}')


            print('\n---------------------------------------------------------\n')

        # if SUPSUP:
        #     model.cache_masks()

        # Now report the numbers for task = 0 upto task = task_id
        print(f'Evaluating on tasks {0} to {task_id}\n', priority = LogPriority.MEDIUM)
        avg_loss = 0
        for t_id in range(task_id+1):
            if t_id != 0:
                test_dl.dataset.set_task(t_id)
            loss, acc = test_loop(model, test_dl, t_id, device = device)
            print(f'Testing Loss for task = {t_id}: {loss}', priority = LogPriority.MEDIUM if task_id != experimenter.cfg.DATASET.NUM_TASKS-1 else LogPriority.STATS )
            print(f'Testing Accuracy for task = {t_id}: {acc}\n', priority = LogPriority.MEDIUM if task_id != experimenter.cfg.DATASET.NUM_TASKS-1 else LogPriority.STATS)
            avg_loss += loss
            task_accs[task_id][t_id] = acc


        experimenter.end_epoch(avg_loss/ (task_id+1))
        print(f'Task Accuracy_t {task_id}: {task_accs}', priority=LogPriority.STATS)

    print(f'Task Accuracy: {task_accs}', priority = LogPriority.STATS)
    avg_acc = task_accs[-1].mean()
    bwt = sum(task_accs[-1]-np.diag(task_accs))/ (task_accs.shape[1]-1)
    print(f'Average Accuracy: {avg_acc}', priority = LogPriority.STATS)
    print(f'BWT: {bwt}', priority = LogPriority.STATS)

    # base_path = '/dccstor/preragar'
    # #base_path = './'
    # pathS = os.path.join(base_path,f'MARK_MODELS/IMAGE/model_{baseline}.pt')
    # # os.makedirs(os.path.split(pathS)[0], exist_ok=True)
    # torch.save(model, pathS)
    return avg_acc, bwt, model


def main(cfg, experimenter, baseline):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print = experimenter.logger.log

    NUM_ITERS = 5
    accs = []
    bwts = []
    for iter in range(NUM_ITERS):
        print(f'Run {iter}', priority = LogPriority.STATS)
        model = MARKModel(cfg, device, 32 if cfg.DATASET.NAME=='Cifar100' else 20, baseline)
        criterion = nn.CrossEntropyLoss().to(device)
        # Different optimizers will be created as and when needed.
        if cfg.DATASET.NAME == 'Cifar100':
            train_dl = get_cifar100_dataloader(num_workers=0, batch_size = cfg.TRAINING.BATCH_SIZE) # Train dl
            test_dl = get_cifar100_dataloader(isTrain=False, num_workers=0, batch_size = cfg.TRAINING.BATCH_SIZE) # Test dl
            val_dl = get_cifar100_dataloader(isTrain=False, isValid=True, num_workers=0, batch_size = cfg.TRAINING.BATCH_SIZE) # Valid dl
        else:
            train_dl = get_cifar100_dataloader(num_workers=0, dset_type = 'img_data', num_tasks = 15, batch_size = cfg.TRAINING.BATCH_SIZE) # Train dl
            test_dl  = get_cifar100_dataloader(isTrain=False, dset_type = 'img_data', num_tasks = 15, num_workers=0, batch_size = cfg.TRAINING.BATCH_SIZE) # Test dl
            val_dl   = get_cifar100_dataloader(isTrain=False, dset_type = 'img_data', num_tasks = 15, isValid=True, num_workers=0, batch_size = cfg.TRAINING.BATCH_SIZE) # Valid dl
        acc, bwt, model = train(model, criterion, train_dl, val_dl, test_dl, experimenter, device, baseline)
        base_path = '/dccstor/preragar'
        #base_path = './'
        pathS = os.path.join(base_path,f'MARK_MODELS/IMAGE/model_{baseline}_{iter}.pt')
        # os.makedirs(os.path.split(pathS)[0], exist_ok=True)
        torch.save(model, pathS)
        accs.append(acc)
        bwts.append(bwt)
    
    accs = np.array(accs); bwts = np.array(bwts);

    print(f'Accuracies: {accs}', priority = LogPriority.STATS)
    print(f'BWTS: {bwts}', priority = LogPriority.STATS)

    print(f'Final Accuracy: {accs.mean()} ( {accs.std()} )', priority = LogPriority.STATS)
    print(f'Final BWT: {bwts.mean()} ( {bwts.std()} )', priority = LogPriority.STATS)



if __name__ == '__main__':
    from experimenter import Experimenter
    cfg_file = 'configs/image_data.yml'
    experimenter = Experimenter(cfg_file)

    # main(experimenter.config, experimenter)
    experimenter.run_experiment()