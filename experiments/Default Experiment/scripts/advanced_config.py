import os
import yaml
from easydict import EasyDict as edict

class AdvancedConfig:

    def save(self, file):
        os.makedirs(os.path.split(file)[0], exist_ok=True)
        yaml.dump(self.config, open(file, 'w'))

    def read_cfg(self, file):
        # Its a json file with comments
        return yaml.safe_load(open(file, 'r'))
        
    def merge_config(self, cfg_dict, base_dict):
        for key in cfg_dict:  
            if key not in base_dict:
                # Strange, raise an error
                raise Exception(f'Key {key} not found in base config')
            if isinstance(cfg_dict[key], dict):
                base_dict[key] = self.merge_config(cfg_dict[key], base_dict[key])
            else:
                base_dict[key] = cfg_dict[key]
        return base_dict

    def __init__(self, file, base_file = 'configs/default.yml') -> None:
        self.default_config = self.read_cfg(base_file)
        self.new_config = self.read_cfg(file)
        self.config = self.merge_config(self.new_config, self.default_config)
        self.config = edict(self.config)
        