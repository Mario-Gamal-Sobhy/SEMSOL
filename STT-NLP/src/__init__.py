import os
import sys
import logging

LOGGING_STR = "[%(asctime)s:%(levelname)s:%(module)s:%(message)s]"
LOG_DIR = "logs"
LOG_FILEPATH = os.path.join(LOG_DIR, "running_log.log")
os.makedirs(LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_STR,
    handlers=[
        logging.FileHandler(LOG_FILEPATH),
        logging.StreamHandler(sys.stdout)
    ]
)

def get_logger(message: str) -> logging.Logger:
    return logging.getLogger(message)
