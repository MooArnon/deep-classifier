import logging
import time

class UTCFormatter(logging.Formatter):
    """
    Custom formatter to force UTC time in logs.
    """
    converter = time.gmtime  # Convert all timestamps to UTC

    def formatTime(self, record, datefmt=None):
        return super().formatTime(record, datefmt)

def get_logger(logger_name: str) -> logging.Logger:
    """
    Returns a configured logger with UTC timestamps, ensuring
    we don't add duplicate handlers.
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False  # Avoid passing logs to higher-level loggers

    # Only add a handler if there aren't any
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = UTCFormatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
