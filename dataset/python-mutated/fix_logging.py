import logging
from ethereum import slogging
orig_getLogger = slogging.SManager.getLogger

def monkey_patched_getLogger(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    orig_class = logging.getLoggerClass()
    result = orig_getLogger(*args, **kwargs)
    logging.setLoggerClass(orig_class)
    return result
slogging.SManager.getLogger = monkey_patched_getLogger