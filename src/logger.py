import logging
import os
from config.config import LOG_DIR

class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

def get_logger(name: str, log_file: str) -> logging.Logger:
    LOG_PATH = LOG_DIR / log_file
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = FlushFileHandler(LOG_PATH, encoding='utf-8', mode='w')
        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s :: %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    return logger