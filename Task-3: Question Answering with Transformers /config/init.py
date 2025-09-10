import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config

def save_config(config: Dict[str, Any], config_path: str = "config/config.yaml") -> None:
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    def _update_nested_dict(original, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in original and isinstance(original[key], dict):
                _update_nested_dict(original[key], value)
            else:
                original[key] = value
        return original
    
    return _update_nested_dict(config.copy(), updates)

# Default configuration
DEFAULT_CONFIG = {
    "model": {
        "checkpoint": "bert-base-cased",
        "save_dir": "bert-finetuned-squad-accelerate",
        "load_from_checkpoint": False,
        "checkpoint_path": ""
    },
    "training": {
        "num_epochs": 3,
        "batch_size": 8,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_steps": 0,
        "max_grad_norm": 1.0,
        "gradient_accumulation_steps": 1
    }
}