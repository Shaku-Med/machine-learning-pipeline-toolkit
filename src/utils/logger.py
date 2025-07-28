import logging
import structlog
from pathlib import Path
from typing import Optional
from src.utils.config import config

def setup_logger(
    name: str = "ml_pipeline",
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> structlog.BoundLogger:
    
    if level is None:
        level = config.get("logging.level", "INFO")
    
    if log_file is None:
        log_file = config.get("logging.file", "logs/ml_pipeline.log")
    
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    logging.basicConfig(
        format=config.get("logging.format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        level=getattr(logging, level.upper()),
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return structlog.get_logger(name)

logger = setup_logger() 