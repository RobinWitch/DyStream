from importlib import import_module
from omegaconf import OmegaConf, DictConfig
import os
from pathlib import Path
import shutil
from lightning.pytorch.utilities import rank_zero_info
import yaml
import argparse
from typing import Any, Dict, Optional

class Config:
    def __init__(self, config_path: str = None, override_args: Dict[str, Any] = None):
        self.config = OmegaConf.create({})
        if config_path:
            self.load_yaml(config_path)
        if override_args:
            self.override_config(override_args)
    
    def load_yaml(self, config_path: str):
        """Load YAML configuration file"""
        self.config = OmegaConf.load(config_path)
    
    def override_config(self, override_args: Dict[str, Any]):
        """Handle command line override arguments"""
        for key, value in override_args.items():
            if '.' in key:
                parts = key.split('.')
                current = self.config
                for part in parts[:-1]:
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                current[parts[-1]] = self._convert_value(value)
            else:
                self.config[key] = self._convert_value(value)
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type"""
        if value.lower() == 'true':
            return True
        elif value.lower() == 'false':
            return False
        elif value.lower() == 'null':
            return None
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return OmegaConf.select(self.config, key, default=default)
    
    def __getattr__(self, name: str) -> Any:
        """Support dot notation access"""
        return self.config[name]
    
    def __getitem__(self, key: str) -> Any:
        """Support dictionary-like access"""
        return self.config[key]
    
    def export_config(self, path: str):
        """Export current configuration to file"""
        OmegaConf.save(self.config, path)

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/motion_gen/0522_3s1d_8_56_clean_prev_audio.yaml',
                      help='Path to config file')
    parser.add_argument('--override', type=str, nargs='+',
                      help='Override config values (key=value)')
    return parser.parse_args()

def load_config(config_path: Optional[str] = None, override_args: Optional[Dict[str, Any]] = None) -> Config:
    """Load configuration"""
    if config_path is None:
        args = parse_args()
        config_path = args.config
        if args.override:
            override_args = {}
            for override in args.override:
                key, value = override.split('=')
                override_args[key.strip()] = value.strip()
    
    return Config(config_path, override_args) 

def instantiate(config: DictConfig, instantiate_module=True):
    """Get arguments from config."""
    module = import_module(config.module_name)
    class_ = getattr(module, config.class_name)
    if instantiate_module:
        init_args = {k: v for k, v in config.items() if k not in ["module_name", "class_name"]}
        return class_(**init_args)
    else:
        return class_

def instantiate_motion_gen(module_name, class_name, cfg, hfstyle=False, **init_args):
    module = import_module(module_name)
    class_ = getattr(module, class_name)
    if hfstyle:
        config_class = class_.config_class
        cfg = config_class(config_obj=cfg)
    return class_(cfg, **init_args)
    
def save_config_and_codes(config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    sanity_check_dir = os.path.join(save_dir, 'sanity_check')
    os.makedirs(sanity_check_dir, exist_ok=True)
    with open(os.path.join(sanity_check_dir, f'{config.exp_name}.yaml'), 'w') as f:
        OmegaConf.save(config.config, f)
    current_dir = Path.cwd()
    # for py_file in current_dir.rglob('*.py'):
    #     dest_path = Path(sanity_check_dir) / py_file.relative_to(current_dir)
    #     dest_path.parent.mkdir(parents=True, exist_ok=True)
    #     shutil.copy(py_file, dest_path)
    
def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    rank_zero_info(f"Total parameters: {total_params:,}")
    rank_zero_info(f"Trainable parameters: {trainable_params:,}")
    rank_zero_info(f"Non-trainable parameters: {(total_params - trainable_params):,}")
    
def load_metrics(file_path):
    metrics = {}
    with open(file_path, "r") as f:
        for line in f:
            key, value = line.strip().split(": ")
            try:
                metrics[key] = float(value)  # Convert to float if possible
            except ValueError:
                metrics[key] = value  # Keep as string if conversion fails
    return metrics