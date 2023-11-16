import threading
import logging
import time
from robot.api import logger

def log_using_robot_api_in_thread():
    if False:
        while True:
            i = 10
    threading.Timer(0.1, log_using_robot_api).start()

def log_using_robot_api():
    if False:
        while True:
            i = 10
    for i in range(100):
        logger.info(str(i))
        time.sleep(0.01)

def log_using_logging_module_in_thread():
    if False:
        print('Hello World!')
    threading.Timer(0.1, log_using_logging_module).start()

def log_using_logging_module():
    if False:
        return 10
    for i in range(100):
        logging.info(str(i))
        time.sleep(0.01)