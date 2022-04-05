import torch
import torch.nn as nn
import numpy as np
from models_marketcl import MARKLSTMModel
from trainer_market import test_loop
from trainer_market import train_fe, train_kb_nonmeta, train_mg_n_clf, update_kb
from dataloaders import get_marketcl_dataloader
from advanced_logger import LogPriority
import pickle

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(model : MARKLSTMModel, criterion, train_dl, experimenter, device):


    print = experimenter.logger.log

    # There are four steps:
    # 1. Train the feature extractor on given task.
    # 2. Train KB if task = 0
    # 3. Train MaskGenerator and Classifier
    # 4. Use Meta-Learning to train the KB, and KBClassifier 
    NUM_DAYS = 53
    NUM_SYMBOLS = 308

    days_accs = np.zeros((4, NUM_DAYS-1))
    days_bwt = np.zeros((4, NUM_DAYS - 1))

    all_freqs = pickle.load(open('datasets/market_cl_freqs.pkl','rb')) 

    for day in range(NUM_DAYS-1): # Last Day only for testing?
        task_accs = np.zeros((4, NUM_SYMBOLS, NUM_SYMBOLS)) 
        print(f'Begin Day {day}', priority = LogPriority.MEDIUM)
        for task_id in range(NUM_SYMBOLS):
            train_dl.dataset.set_day(day)
            train_dl.dataset.set_symbol(task_id)

            freqs = all_freqs[day][task_id]
            criterions = [None, None, None, None]
            for l in freqs:
                freq = freqs[l]
                print([v/(sum(freq.values())) for v in freq.values()])
                class_weights = [1/f if f != 0 else 0 for f in freq.values()]
                class_weights = torch.tensor([x/sum(class_weights) for x in class_weights], device = device)
                criterion = nn.CrossEntropyLoss(weight = class_weights)
                criterions[l] = criterion

            print('Frequencies for Current Day: ', freqs)
            # Step 1: First train the feature extractor on task 
            # Note: 50 epochs are more than enough

            # '''
            print(f'Training Feature Extractor on task id {task_id}')
            loss, acc = train_fe(model, train_dl, criterions, task_id, device = device, lr = float(experimenter.config.TRAINING.LR.FE), num_epochs = experimenter.config.TRAINING.EPOCHS.FE)
            print(f'Feature Extraction Train Loss {loss} & Accuracy: {acc}')
            
            print('\n---------------------------------------------------------\n')

            # Step 2: Now if task is 0, train the KB without Meta-Learning
            if task_id == 0:
                print(f'Training Initial Knowledge Base Weights')
                loss, acc = train_kb_nonmeta(model, train_dl, criterions, task_id, lr = float(experimenter.config.TRAINING.LR.INIT_KB), device = device, num_epochs = experimenter.config.TRAINING.EPOCHS.INIT_KB)
                print(f'Initial KB Validation Loss {loss} & Accuracy: {acc}')

                print('\n---------------------------------------------------------\n')

            # Step 3: Now train MaskGenerator and Classifier
            print(f'Training Mask Generator and Classifier')
            loss, acc = train_mg_n_clf(model, train_dl, criterions, task_id, lr = float(experimenter.config.TRAINING.LR.MG_N_C), device = device, num_epochs = experimenter.config.TRAINING.EPOCHS.MG_N_C)
            print(f'Mask Generation and Classifier Training Loss {loss} & Accuracy: {acc}')

            print('\n---------------------------------------------------------\n')
            
            # '''
            print('Updating KB')
            print('Before',sum([x.sum() for x in model.kb.parameters()]))
            update_kb(model, train_dl, train_dl, criterions, task_id, device=device)
            print(f'Update KB {loss} & Accuracy: {acc}')
            print('After',sum([x.sum() for x in model.kb.parameters()]))

            print('\n---------------------------------------------------------\n')
            # '''

            # Stage 5: Fine-Tune Mask Generator and Final Classifier
            print(f'Fine-Tune Mask Generator and Classifier')
            train_mg_n_clf(model, train_dl, criterions, task_id, lr = float(experimenter.config.TRAINING.LR.FINETUNE_MG_N_C), num_epochs = experimenter.config.TRAINING.EPOCHS.FINETUNE_MG_N_C, device = device)
            print(f'Fine-Tuning Mask Generation and Classifier Training Loss {loss} & Accuracy: {acc}')


            print('\n---------------------------------------------------------\n')


            # Now report the numbers for task = 0 upto task = task_id
            print(f'Evaluating on tasks {0} to {task_id}\n', priority = LogPriority.MEDIUM)
            avg_loss = 0
            for t_id in range(task_id+1):
                train_dl.dataset.set_day(day+1)
                train_dl.dataset.set_symbol(t_id)
                loss, acc, f1s = test_loop(model, train_dl, t_id, criterions, device = device)
                print(f'Testing Loss for task = {t_id}: {loss}', priority = LogPriority.MEDIUM if task_id != experimenter.cfg.DATASET.NUM_TASKS-1 else LogPriority.STATS )
                print(f'Testing Accuracy for task = {t_id}: {acc}', priority = LogPriority.MEDIUM if task_id != experimenter.cfg.DATASET.NUM_TASKS-1 else LogPriority.STATS)
                print(f'Testing F1s for task = {t_id}: {f1s}\n', priority = LogPriority.MEDIUM if task_id != experimenter.cfg.DATASET.NUM_TASKS-1 else LogPriority.STATS)
                
                avg_loss += loss
                for i, ac in acc.items():
                    task_accs[i][task_id][t_id] = ac

        print(f'Day {day} complete', priority = LogPriority.STATS)
        for i in range(4):
            print(f'Label {i}', priority = LogPriority.STATS)
            ta = task_accs[i]
            avg_acc = ta[-1].mean()
            bwt = sum(ta[-1]-np.diag(ta))/ (ta.shape[1]-1)
            print(f'Average Accuracy: {avg_acc}', priority = LogPriority.STATS)
            print(f'BWT: {bwt}', priority = LogPriority.STATS)
            days_accs[i][day] =  avg_acc
            days_bwt[i][day] = bwt
    # Note this returns only for day 53 and 3rd label
    return days_bwt.mean(axis=1), days_bwt.mean(axis=1)


def main(cfg, experimenter):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # device = 'cpu'
    print = experimenter.logger.log

    NUM_ITERS = 5
    accs = []
    bwts = []
    for iter in range(NUM_ITERS):
        print(f'Run {iter}', priority = LogPriority.STATS)
        model = MARKLSTMModel(cfg, device)
        criterion = nn.CrossEntropyLoss().to(device)
        # Different optimizers will be created as and when needed.
    
        train_dl = get_marketcl_dataloader(batch_size = cfg.TRAINING.BATCH_SIZE) # Train dl

        acc, bwt = train(model, criterion, train_dl, experimenter, device)
        accs.append(acc)
        bwts.append(bwt)
    
    accs = np.array(accs); bwts = np.array(bwts);

    print(f'Accuracies: {accs}', priority = LogPriority.STATS)
    print(f'BWTS: {bwts}', priority = LogPriority.STATS)

    print(f'Final Accuracy: {accs.mean()} ( {accs.std()} )', priority = LogPriority.STATS)
    print(f'Final BWT: {bwts.mean()} ( {bwts.std()} )', priority = LogPriority.STATS)



if __name__ == '__main__':
    from experimenter import Experimenter
    cfg_file = 'configs/market1.yml'#sys.argv[1]
    experimenter = Experimenter(cfg_file)
    main(experimenter.config, experimenter)
    # experimenter.run_experiment()