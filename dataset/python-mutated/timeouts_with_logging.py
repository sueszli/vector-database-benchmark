import logging
import time
from robot.api import logger
from robot.output.pyloggingconf import RobotHandler
for handler in logging.getLogger().handlers:
    if isinstance(handler, RobotHandler):
        handler.format = lambda record: record.getMessage()
MSG = 'A rather long message that is slow to write on the disk. ' * 10000

def rf_logger():
    if False:
        print('Hello World!')
    _log_a_lot(logger.info)

def python_logger():
    if False:
        while True:
            i = 10
    _log_a_lot(logging.info)

def _log_a_lot(info):
    if False:
        print('Hello World!')
    msg = MSG
    sleep = time.sleep
    current = time.time
    end = current() + 1
    while current() < end:
        info(msg)
        sleep(0)
    raise AssertionError('Execution should have been stopped by timeout.')