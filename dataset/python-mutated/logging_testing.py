import logging

class CustomFormatter(logging.Formatter):
    grey = '\x1b[38;20m'
    yellow = '\x1b[33;20m'
    red = '\x1b[31;20m'
    bold_red = '\x1b[31;1m'
    reset = '\x1b[0m'
    format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)'
    FORMATS = {logging.DEBUG: grey + format + reset, logging.INFO: grey + format + reset, logging.WARNING: yellow + format + reset, logging.ERROR: red + format + reset, logging.CRITICAL: bold_red + format + reset}

    def format(self, record):
        if False:
            print('Hello World!')
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

def testing(logger):
    if False:
        return 10
    logger.debug('debug message')
    logger.info('info message')
    logger.warning('warning message')
    logger.error('error message')
    logger.critical('critical message')
if __name__ == '__main__':
    from logging_testing import *
    logger = logging.getLogger('My_app')
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    testing(logger)