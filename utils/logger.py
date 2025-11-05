"""
Logging utilities for ComfyUI Motion Transfer.

Provides consistent logging across all nodes with configurable levels and formatting.
"""

import logging
import time
from functools import wraps
from typing import Callable, Any

# Global logger instance
_logger = None


def setup_logging(level: str = "INFO", format_string: str = None) -> logging.Logger:
    """
    Setup logging configuration for Motion Transfer.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string (optional)

    Returns:
        Configured logger instance
    """
    global _logger

    if _logger is not None:
        return _logger

    # Create logger
    _logger = logging.getLogger("MotionTransfer")
    _logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers
    if _logger.handlers:
        return _logger

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    # Create formatter
    if format_string is None:
        format_string = "[%(name)s] [%(levelname)s] %(message)s"

    formatter = logging.Formatter(format_string)
    console_handler.setFormatter(formatter)

    # Add handler to logger
    _logger.addHandler(console_handler)

    # Prevent propagation to root logger (avoid double logging)
    _logger.propagate = False

    return _logger


def get_logger() -> logging.Logger:
    """
    Get the Motion Transfer logger instance.

    Returns:
        Logger instance (creates with default settings if not already setup)
    """
    global _logger

    if _logger is None:
        return setup_logging()

    return _logger


def log_performance(func: Callable) -> Callable:
    """
    Decorator to log function execution time.

    Usage:
        @log_performance
        def my_function():
            ...

    Args:
        func: Function to wrap

    Returns:
        Wrapped function that logs execution time
    """

    @wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        logger = get_logger()
        start_time = time.time()

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time

            # Log at DEBUG level
            logger.debug(
                f"{func.__name__} completed in {elapsed:.2f}s"
            )

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {elapsed:.2f}s: {str(e)}"
            )
            raise

    return wrapper


class LogContext:
    """
    Context manager for logging operations with timing.

    Usage:
        with LogContext("Processing frame"):
            # do work
            pass

        Output: [MotionTransfer] [INFO] Processing frame... (started)
                [MotionTransfer] [INFO] Processing frame completed in 1.23s
    """

    def __init__(self, operation: str, level: str = "INFO"):
        """
        Initialize log context.

        Args:
            operation: Description of the operation
            level: Log level to use
        """
        self.operation = operation
        self.level = level
        self.logger = get_logger()
        self.start_time = None

    def __enter__(self):
        """Start timing and log operation start."""
        self.start_time = time.time()
        log_func = getattr(self.logger, self.level.lower())
        log_func(f"{self.operation}... (started)")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Log operation completion with timing."""
        elapsed = time.time() - self.start_time
        log_func = getattr(self.logger, self.level.lower())

        if exc_type is None:
            log_func(f"{self.operation} completed in {elapsed:.2f}s")
        else:
            self.logger.error(
                f"{self.operation} failed after {elapsed:.2f}s: {exc_val}"
            )

        return False  # Don't suppress exceptions


# Initialize logger on module import
setup_logging()
