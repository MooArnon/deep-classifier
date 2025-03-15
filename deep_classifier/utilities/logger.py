##########
# Import #
##############################################################################

import logging
from datetime import datetime, timezone
import pytz

###########
# Classes #
##############################################################################

# Define a custom formatter that uses UTC time
class UTCFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        # Use timezone-aware datetime
        utc_time = datetime.fromtimestamp(record.created, tz=timezone.utc)
        if datefmt:
            return utc_time.strftime(datefmt)
        else:
            return utc_time.strftime('%Y-%m-%d %H:%M:%S')

############
# Function #
##############################################################################

def get_logger(logger_name: str):
    """
    Creates and returns a configured logger object that prints logs to the terminal.

    Returns:
        logging.Logger: The configured logger object.
    """
    # Set up the logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)  # Set the desired logging level

    # Create a stream handler that logs debug and higher level messages to the terminal
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    formatter = UTCFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(handler)

    return logger
