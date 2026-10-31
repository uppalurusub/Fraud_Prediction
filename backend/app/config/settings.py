# app/utils/logger.py

import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path
from loguru import logger

PROJECT_ROOT = Path(__file__).resolve().parents[3]
BACKEND_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BACKEND_DIR / "data"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"

import os
os.makedirs(DATA_DIR,exist_ok=True)
os.makedirs(ARTIFACT_DIR,exist_ok=True)
os.makedirs(RESULTS_DIR,exist_ok=True)
os.makedirs(LOG_DIR,exist_ok=True)



# ------------------------------------------------------------------------------
# SETTINGS CLASS
# ------------------------------------------------------------------------------

class Settings:

    data_dir: str = DATA_DIR
    #print("data_dir: ",data_dir)

    artifact_dir: str = ARTIFACT_DIR
    #print("artifact_dir: ",artifact_dir)

    results_dir: str = RESULTS_DIR
    #print("results_dir: ",results_dir)

    log_dir: str = LOG_DIR
    #print("log_dir: ",log_dir)
    # ------------------------------------------------------------------------------
    # GET LOGGER INSTANCE
    # ------------------------------------------------------------------------------

    def get_logger(name: str = __name__):
        """
        Return bound logger instance

        Usage:
            logger = get_logger(__name__)
            logger.info("Application started")
        """

        return logger.bind(module=name)


# ------------------------------------------------------------------------------
# LOGGER CLASS
# ------------------------------------------------------------------------------

class LoggerManager:

    @staticmethod
    def info(message: str):
        logger.info(message)

    @staticmethod
    def warning(message: str):
        logger.warning(message)

    @staticmethod
    def error(message: str):
        logger.error(message)

    @staticmethod
    def debug(message: str):
        logger.debug(message)

    @staticmethod
    def critical(message: str):
        logger.critical(message)


# ------------------------------------------------------------------------------
# REMOVE DEFAULT LOGGER
# ------------------------------------------------------------------------------

logger.remove()

# ------------------------------------------------------------------------------
# CONSOLE LOGGER
# ------------------------------------------------------------------------------

logger.add(
    sys.stdout,
    level="INFO",
    colorize=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
)

# ------------------------------------------------------------------------------
# INFO FILE LOGGER
# ------------------------------------------------------------------------------

logger.add(
    f"{LOG_DIR}/application.log",
    rotation="10 MB",
    retention="30 days",
    compression="zip",
    level="INFO",
    enqueue=True,
    backtrace=True,
    diagnose=True,
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
)

# ------------------------------------------------------------------------------
# ERROR FILE LOGGER
# ------------------------------------------------------------------------------

logger.add(
    f"{LOG_DIR}/error.log",
    rotation="5 MB",
    retention="60 days",
    compression="zip",
    level="ERROR",
    enqueue=True,
    backtrace=True,
    diagnose=True,
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )
)

settings = Settings()