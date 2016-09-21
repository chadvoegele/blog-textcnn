import logging

from . import cnn
from . import data
from . import majority

__all__ = ['cnn', 'data']

logging.basicConfig(level=logging.DEBUG, format="%(threadName)s - %(asctime)s - %(filename)s - %(funcName)s: %(message)s")
logging.captureWarnings(True)
