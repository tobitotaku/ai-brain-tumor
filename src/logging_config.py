"""
Enhanced Logging Configuration
==============================
Centralized logging setup with console and file output for better visibility.

Features:
- Colored console output with timestamps
- File logging with rotation
- Progress tracking with custom formatters
- Real-time metrics display
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Add colors to console output for better readability."""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{log_color}{record.levelname}{self.RESET}"
        return super().format(record)


def setup_logging(
    log_dir: str = 'logs',
    log_name: Optional[str] = None,
    level: int = logging.INFO,
    use_colors: bool = True
) -> logging.Logger:
    """
    Configure logging with console and file handlers.
    
    Parameters:
        log_dir: Directory to store log files
        log_name: Name for log file (default: training_TIMESTAMP)
        level: Logging level
        use_colors: Whether to use colored console output
    
    Returns:
        Configured logger instance
    """
    
    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    # Generate log filename if not provided
    if log_name is None:
        log_name = f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    
    log_file = log_path / f'{log_name}.log'
    
    # Create logger
    logger = logging.getLogger('training_pipeline')
    logger.setLevel(level)
    logger.propagate = False
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    if use_colors:
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%H:%M:%S'
        )
    else:
        console_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=10*1024*1024,  # 10 MB
        backupCount=5
    )
    file_handler.setLevel(level)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Log initial message with file location
    logger.info(f"Logging initialized | File: {log_file}")
    
    return logger


def log_section(logger: logging.Logger, title: str, level: int = logging.INFO):
    """
    Log a formatted section header.
    
    Parameters:
        logger: Logger instance
        title: Section title
        level: Logging level
    """
    logger.log(level, f"\n{'='*70}")
    logger.log(level, f"{title}")
    logger.log(level, f"{'='*70}")


def log_subsection(logger: logging.Logger, title: str, level: int = logging.INFO):
    """
    Log a formatted subsection header.
    
    Parameters:
        logger: Logger instance
        title: Subsection title
        level: Logging level
    """
    logger.log(level, f"\n{'\u250c'} {title}")


def log_metrics(logger: logging.Logger, metrics: dict, indent: str = "  "):
    """
    Log metrics in a formatted table.
    
    Parameters:
        logger: Logger instance
        metrics: Dictionary of metric names and values
        indent: Indentation string
    """
    for key, value in metrics.items():
        if isinstance(value, float):
            logger.info(f"{indent}{key:.<30} {value:.4f}")
        else:
            logger.info(f"{indent}{key:.<30} {value}")


def log_fold_progress(
    logger: logging.Logger,
    fold_num: int,
    total_folds: int,
    elapsed: Optional[float] = None,
    metrics: Optional[dict] = None
):
    """
    Log cross-validation fold progress.
    
    Parameters:
        logger: Logger instance
        fold_num: Current fold number
        total_folds: Total number of folds
        elapsed: Elapsed time in seconds (optional)
        metrics: Metrics dictionary (optional)
    """
    msg = f"Fold {fold_num}/{total_folds}"
    if elapsed is not None:
        msg += f" | {elapsed:.1f}s"
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f" | {key}: {value:.3f}"
    
    logger.info(msg)


def get_logger(name: str = 'training_pipeline') -> logging.Logger:
    """
    Get existing logger instance.
    
    Parameters:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)
