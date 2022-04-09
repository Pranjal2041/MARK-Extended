# Experimenter Class is responsible for mainly four things:
# 1. Configuration - Done
# 2. Logging using the AdvancedLogger class - Almost Done
# 3. Model Handling, including loading and saving models - Done(Upgrades Left)
# 4. Running Different Variants Paralelly/Sequentially of experiments
# 5. Combining frcnn training followed by bilateral training and final froc calculation - Done
# 6. Version Control

from advanced_config import AdvancedConfig
from advanced_logger import AdvancedLogger, LogPriority
import os
from os.path import join
import torch
import argparse
from train import main as TRAIN_MARK
from utils import create_backup
from torch.utils.tensorboard import SummaryWriter

class Experimenter:

    def __init__(self, cfg_file, BASE_DIR = 'experiments'):
        self.cfg_file = cfg_file        
        self.curr_epoch = 0
        self.curr_mode = 'MARK'
        self.con = AdvancedConfig(cfg_file)
        self.config = self.con.config
        self.cfg = self.config
        self.exp_dir = join(BASE_DIR,self.config.EXP_NAME)
        os.makedirs(self.exp_dir, exist_ok=True)
        self.con.save(join(self.exp_dir,'config.yml'))
    
        self.logger = AdvancedLogger(self.exp_dir)
        self.logger.log('Experiment:',self.config.EXP_NAME,priority = LogPriority.STATS)
        self.logger.log('Experiment Description:', self.config.EXP_DESC, priority = LogPriority.STATS)
        self.logger.log('Config File:',self.cfg_file, priority = LogPriority.STATS)
        self.logger.log('Experiment started', priority = LogPriority.LOW)
        self.losses = dict()

        self.writer = SummaryWriter(join(self.exp_dir,'tensor_logs'))

        create_backup(backup_dir=join(self.exp_dir,'scripts'))

    def log(self, *args, **kwargs):
        self.logger.log(*args, **kwargs)


    def init_losses(self,mode):
        if mode == 'MARK':
            self.losses['mark_loss'] = []

    def start_epoch(self):
        self.curr_epoch += 1
        self.logger.log('Epoch:',self.curr_epoch, priority = LogPriority.MEDIUM)

    def end_epoch(self, loss):
        if self.curr_mode == 'MARK':
            self.losses['mark_loss'].append(loss)
            self.best_loss = min(self.losses['mark_loss'])
        self.writer.add_scalar(f"{self.curr_mode}/Loss/Valid", loss, self.curr_epoch)


    def load_models(self, model_path = '', **kwargs):
        if model_path == '': return
        model_path = join(self.exp_dir,'{k}_model.pth')
        for k,v in kwargs.items():
            v.load_state_dict(torch.load(model_path.format(k=k)))
            self.logger.log(f'Loaded {k} model', priority = LogPriority.LOW)

    def save_models(self, saveAll = True, **kwargs):
        if self.curr_mode == 'MARK':
            self.logger.log('Saving MARK Model', priority = LogPriority.LOW)
            model_files = {k:join(self.exp_dir,'mark_models',f'{k}_model.pth') for k,v in kwargs.items()}
            SAVE = self.best_loss == self.losses['mark_loss'][-1]
        if SAVE:
            for k,v in model_files.items():
                os.makedirs(os.path.split(v)[0], exist_ok=True)
                torch.save(kwargs[k].state_dict(),v)
                self.logger.log(f'Saved {k} model', priority = LogPriority.LOW)
        if saveAll:
            for k,v in model_files.items():
                os.makedirs(os.path.split(v)[0], exist_ok=True)
                torch.save(kwargs[k].state_dict(),v.replace('.pth',f'_{self.curr_epoch}.pth'))
                self.logger.log(f'Saved {k} model', priority = LogPriority.LOW)


    # Note: Here epoch is not the same as actual epoch but rather indicates 
    # A complete pipeline complete
    # Currently Only Architecture Search is supported
    def run_experiment(self):

        # First Determine the mode of running the experiment
        mode = self.config.MODE
        self.init_losses(mode)
        self.curr_mode = 'MARK'
        self.curr_epoch = -1
        self.best_loss = 999999
        if mode == 'MARK':
            TRAIN_MARK(self.config, self)

        self.logger.log(f'Best Loss: {self.best_loss}', priority= LogPriority.STATS)
        self.logger.log('Experiment Training and Generation Ended', priority = LogPriority.MEDIUM)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg_file', type=str, default='configs/debug.yml')
    args = parser.parse_args()
    exp = Experimenter(args.cfg_file)
    exp.run_experiment()