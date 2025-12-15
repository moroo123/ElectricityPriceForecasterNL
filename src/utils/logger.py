import logging
import sys
from pathlib import Path

def setup_logger(name: str = "forecaster", level: int = logging.INFO) -> logging.Logger:
    """
    Setup standard logger with console and file handlers.
    
    Parameters
    ----------
    name : str
        Name of the logger
    level : int
        Logging level
        
    Returns
    -------
    logging.Logger
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Prevent adding handlers multiple times
    if logger.hasHandlers():
        return logger
        
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "app.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger
