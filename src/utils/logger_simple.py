import logging
from pathlib import Path
from typing import Optional
from src.utils.config import config

def setup_logger(
    name: str = "ml_pipeline",
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    
    if level is None:
        level = config.get("logging.level", "INFO")
    
    if log_file is None:
        log_file = config.get("logging.file", "logs/ml_pipeline.log")
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        formatter = logging.Formatter(
            config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

logger = setup_logger() 