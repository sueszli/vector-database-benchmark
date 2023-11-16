import logging
import os
import ray
from ray._private.ray_constants import LOGGER_FORMAT, LOGGER_LEVEL

class DatasetLogger:
    """Logger for Ray Datasets which writes logs to a separate log file
    at `DatasetLogger.DEFAULT_DATASET_LOG_PATH`. Can optionally turn off
    logging to stdout to reduce clutter (but always logs to the aformentioned
    Datasets-specific log file).

    After initialization, always use the `get_logger()` method to correctly
    set whether to log to stdout. Example usage:
    ```
    logger = DatasetLogger(__name__)
    logger.get_logger().info("This logs to file and stdout")
    logger.get_logger(log_to_stdout=False).info("This logs to file only)
    logger.get_logger().warning("Can call the usual Logger methods")
    ```
    """
    DEFAULT_DATASET_LOG_PATH = 'logs/ray-data.log'

    def __init__(self, log_name: str):
        if False:
            i = 10
            return i + 15
        'Initialize DatasetLogger for a given `log_name`.\n\n        Args:\n            log_name: Name of logger (usually passed into `logging.getLogger(...)`)\n        '
        self.log_name = log_name
        self._logger = None

    def _initialize_logger(self) -> logging.Logger:
        if False:
            print('Hello World!')
        'Internal method to initialize the logger and the extra file handler\n        for writing to the Dataset log file. Not intended (nor necessary)\n        to call explicitly. Assumes that `ray.init()` has already been called prior\n        to calling this method; otherwise raises a `ValueError`.'
        stdout_logger = logging.getLogger(self.log_name)
        stdout_logger.setLevel(LOGGER_LEVEL.upper())
        logger = logging.getLogger(f'{self.log_name}.logfile')
        logger.setLevel(LOGGER_LEVEL.upper())
        global_node = ray._private.worker._global_node
        if global_node is not None:
            session_dir = global_node.get_session_dir_path()
            datasets_log_path = os.path.join(session_dir, DatasetLogger.DEFAULT_DATASET_LOG_PATH)
            file_log_formatter = logging.Formatter(fmt=LOGGER_FORMAT)
            file_log_handler = logging.FileHandler(datasets_log_path)
            file_log_handler.setLevel(LOGGER_LEVEL.upper())
            file_log_handler.setFormatter(file_log_formatter)
            logger.addHandler(file_log_handler)
        return logger

    def get_logger(self, log_to_stdout: bool=True) -> logging.Logger:
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns the underlying Logger, with the `propagate` attribute set\n        to the same value as `log_to_stdout`. For example, when\n        `log_to_stdout = False`, we do not want the `DatasetLogger` to\n        propagate up to the base Logger which writes to stdout.\n\n        This is a workaround needed due to the DatasetLogger wrapper object\n        not having access to the log caller\'s scope in Python <3.8.\n        In the future, with Python 3.8 support, we can use the `stacklevel` arg,\n        which allows the logger to fetch the correct calling file/line and\n        also removes the need for this getter method:\n        `logger.info(msg="Hello world", stacklevel=2)`\n        '
        if self._logger is None:
            self._logger = self._initialize_logger()
        self._logger.propagate = log_to_stdout
        return self._logger