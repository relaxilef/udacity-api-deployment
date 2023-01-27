"""
Contains the global definitions for the project.
"""

import logging
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    datefmt="%d-%m-%y %H:%M:%S",
)
logger = logging.getLogger(__name__)
