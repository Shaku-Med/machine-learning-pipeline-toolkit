import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

class ConfigManager:
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = Path(config_path)
        self._config = None
        load_dotenv()
        self._load_config()
    
    def _load_config(self):
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as file:
            self._config = yaml.safe_load(file)
        
        self._override_with_env()
    
    def _override_with_env(self):
        env_mappings = {
            'APP_ENV': 'app.environment',
            'LOG_LEVEL': 'logging.level',
            'API_HOST': 'deployment.host',
            'API_PORT': 'deployment.port',
            'MLFLOW_TRACKING_URI': 'mlflow.tracking_uri'
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value:
                self._set_nested_value(config_path, env_value)
    
    def _set_nested_value(self, path: str, value: Any):
        keys = path.split('.')
        config = self._config
        for key in keys[:-1]:
            config = config.setdefault(key, {})
        config[keys[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        keys = key.split('.')
        value = self._config
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_all(self) -> Dict[str, Any]:
        return self._config.copy()
    
    def update(self, key: str, value: Any):
        self._set_nested_value(key, value)

config = ConfigManager() 