"""
Unified logging utilities for PRISM project.

Provides consistent logging configuration across all modules.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color support for terminal output."""

    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'  # Reset
    }

    def format(self, record):
        """Format log record with colors if terminal supports it."""
        if sys.stdout.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"

        return super().format(record)


def setup_logger(
        name: str,
        level: int = logging.INFO,
        log_file: Optional[Path] = None,
        use_color: bool = True
) -> logging.Logger:
    """
    Setup and configure a logger.

    Args:
        name: Logger name (typically __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional file path to write logs
        use_color: Use colored output for terminal

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Training started")
    """
    logger = logging.getLogger(name)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)

    if use_color:
        console_format = ColoredFormatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        console_format = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    # File handler (if specified)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)

        file_format = logging.Formatter(
            '[%(asctime)s] %(name)s - %(levelname)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get or create a logger with default configuration.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Logger instance

    Example:
        >>> from src.utils.logger import get_logger
        >>> logger = get_logger(__name__)
    """
    return setup_logger(name)


class ProgressLogger:
    """
    Logger for progress tracking with statistics.

    Example:
        >>> progress = ProgressLogger("Training", total=100)
        >>> for i in range(100):
        >>>     progress.update(i+1, loss=0.5, acc=0.9)
    """

    def __init__(self, task_name: str, total: int, logger: Optional[logging.Logger] = None):
        """
        Initialize progress logger.

        Args:
            task_name: Name of the task being tracked
            total: Total number of steps
            logger: Logger instance (creates new if None)
        """
        self.task_name = task_name
        self.total = total
        self.logger = logger or get_logger(__name__)
        self.current = 0

    def update(self, step: int, **metrics):
        """
        Update progress with optional metrics.

        Args:
            step: Current step number
            **metrics: Key-value pairs of metrics to log
        """
        self.current = step
        progress_pct = (step / self.total) * 100

        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                  for k, v in metrics.items()])

        msg = f"{self.task_name} [{step}/{self.total}] ({progress_pct:.1f}%)"
        if metrics_str:
            msg += f" | {metrics_str}"

        self.logger.info(msg)

    def complete(self, **final_metrics):
        """
        Log completion message with final metrics.

        Args:
            **final_metrics: Final metrics to log
        """
        metrics_str = " | ".join([f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}"
                                  for k, v in final_metrics.items()])

        msg = f"{self.task_name} completed"
        if metrics_str:
            msg += f" | {metrics_str}"

        self.logger.info(msg)


def log_section(logger: logging.Logger, title: str, width: int = 80):
    """
    Log a section header for better visual separation.

    Args:
        logger: Logger instance
        title: Section title
        width: Width of the separator line

    Example:
        >>> log_section(logger, "Stage 1 Training")
    """
    logger.info("=" * width)
    logger.info(title.center(width))
    logger.info("=" * width)


def log_config(logger: logging.Logger, config: dict, title: str = "Configuration"):
    """
    Log configuration parameters in a readable format.

    Args:
        logger: Logger instance
        config: Configuration dictionary
        title: Title for the config section

    Example:
        >>> log_config(logger, {"lr": 0.001, "batch_size": 16})
    """
    logger.info("-" * 80)
    logger.info(title)
    logger.info("-" * 80)
    for key, value in config.items():
        logger.info(f"  {key}: {value}")
    logger.info("-" * 80)


def log_metrics(logger: logging.Logger, metrics: dict, prefix: str = ""):
    """
    Log metrics in a consistent format.

    Args:
        logger: Logger instance
        metrics: Dictionary of metrics
        prefix: Optional prefix for metric names

    Example:
        >>> log_metrics(logger, {"loss": 0.5, "accuracy": 0.9}, prefix="val")
    """
    for key, value in metrics.items():
        metric_name = f"{prefix}_{key}" if prefix else key
        if isinstance(value, float):
            logger.info(f"  {metric_name}: {value:.4f}")
        else:
            logger.info(f"  {metric_name}: {value}")


class DualLogger:
    """
    Logger that writes to both console and file simultaneously.

    Useful for long-running training scripts where you want
    both real-time console output and a persistent log file.

    Example:
        >>> logger = DualLogger("training", log_dir="logs")
        >>> logger.info("Training started")
    """

    def __init__(self, name: str, log_dir: str = "logs", level: int = logging.INFO):
        """
        Initialize dual logger.

        Args:
            name: Base name for log files
            log_dir: Directory to store log files
            level: Logging level
        """
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create log file with timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{name}_{timestamp}.log"

        self.logger = setup_logger(
            name=f"{name}_dual",
            level=level,
            log_file=log_file,
            use_color=True
        )

        self.logger.info(f"Logging to file: {log_file}")

    def debug(self, msg: str):
        """Log debug message."""
        self.logger.debug(msg)

    def info(self, msg: str):
        """Log info message."""
        self.logger.info(msg)

    def warning(self, msg: str):
        """Log warning message."""
        self.logger.warning(msg)

    def error(self, msg: str):
        """Log error message."""
        self.logger.error(msg)

    def critical(self, msg: str):
        """Log critical message."""
        self.logger.critical(msg)


# Convenience function for quick logger setup
def quick_setup(name: str = "prism", level: str = "INFO") -> logging.Logger:
    """
    Quick setup for basic logging needs.

    Args:
        name: Logger name
        level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger

    Example:
        >>> logger = quick_setup()
        >>> logger.info("Quick logging setup complete")
    """
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }

    return setup_logger(name, level=level_map.get(level.upper(), logging.INFO))


# Module-level logger for utility functions
_module_logger = None


def get_module_logger() -> logging.Logger:
    """Get module-level logger for utility functions."""
    global _module_logger
    if _module_logger is None:
        _module_logger = setup_logger("prism.utils")
    return _module_logger