import logging
import sys


def set_basic_config(logger):
    if not len(logger.handlers):
        logger_handler = logging.StreamHandler(sys.stdout)
        logger_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT, None))
        logger.addHandler(logger_handler)
