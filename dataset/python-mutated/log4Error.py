import logging
logger = logging.getLogger(__name__)

def outputUserMessage(errMsg, fixMsg=None):
    if False:
        print('Hello World!')
    logger.error(f'\n\n****************************Usage Error************************\n' + errMsg)
    if fixMsg:
        logger.error(f'\n\n**************************How to fix***********************\n' + fixMsg)
    logger.error(f'\n\n****************************Call Stack*************************')

def invalidInputError(condition, errMsg, fixMsg=None):
    if False:
        i = 10
        return i + 15
    if not condition:
        outputUserMessage(errMsg, fixMsg)
        raise RuntimeError(errMsg)

def invalidOperationError(condition, errMsg, fixMsg=None, cause=None):
    if False:
        while True:
            i = 10
    if not condition:
        outputUserMessage(errMsg, fixMsg)
        if cause:
            raise cause
        else:
            raise RuntimeError(errMsg)

class MuteHFLogger:

    def __init__(self, logger, speak_level=logging.ERROR) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.logger = logger
        self.speak_level = speak_level
        self.old_level = logger.getEffectiveLevel()

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self.logger.setLevel(self.speak_level)

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            return 10
        self.logger.setLevel(self.old_level)