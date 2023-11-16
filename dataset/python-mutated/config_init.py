import copy
import logging
import os
import sys

def initialize_logging():
    if False:
        i = 10
        return i + 15
    loggers = {}
    logger_env_vars = {'FEATURETOOLS_LOG_LEVEL': 'featuretools', 'FEATURETOOLS_ES_LOG_LEVEL': 'featuretools.entityset', 'FEATURETOOLS_BACKEND_LOG_LEVEL': 'featuretools.computation_backend'}
    for (logger_env, logger) in logger_env_vars.items():
        log_level = os.environ.get(logger_env, None)
        if log_level is not None:
            loggers[logger] = log_level
    loggers.setdefault('featuretools', 'info')
    loggers.setdefault('featuretools.computation_backend', 'info')
    loggers.setdefault('featuretools.entityset', 'info')
    fmt = '%(asctime)-15s %(name)s - %(levelname)s    %(message)s'
    out_handler = logging.StreamHandler(sys.stdout)
    err_handler = logging.StreamHandler(sys.stdout)
    out_handler.setFormatter(logging.Formatter(fmt))
    err_handler.setFormatter(logging.Formatter(fmt))
    err_levels = ['WARNING', 'ERROR', 'CRITICAL']
    for (name, level) in list(loggers.items()):
        LEVEL = getattr(logging, level.upper())
        logger = logging.getLogger(name)
        logger.setLevel(LEVEL)
        for _handler in logger.handlers:
            logger.removeHandler(_handler)
        if level in err_levels:
            logger.addHandler(err_handler)
        else:
            logger.addHandler(out_handler)
        logger.propagate = False
initialize_logging()

class Config:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._data = {}
        self.set_to_default()

    def set_to_default(self):
        if False:
            return 10
        PWD = os.path.dirname(__file__)
        primitive_data_folder = os.path.join(PWD, 'primitives/data')
        self._data = {'primitive_data_folder': primitive_data_folder}

    def get(self, key):
        if False:
            while True:
                i = 10
        return copy.deepcopy(self._data[key])

    def get_all(self):
        if False:
            while True:
                i = 10
        return copy.deepcopy(self._data)

    def set(self, values):
        if False:
            for i in range(10):
                print('nop')
        self._data.update(values)
config = Config()