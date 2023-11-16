import gc
import logging
import platform
logger = logging.getLogger(__name__)

def gc_set_threshold():
    if False:
        print('Hello World!')
    '\n    Reduce number of GC runs to improve performance (explanation video)\n    https://www.youtube.com/watch?v=p4Sn6UcFTOU\n\n    '
    if platform.python_implementation() == 'CPython':
        gc.set_threshold(50000, 500, 1000)
        logger.debug('Adjusting python allocations to reduce GC runs')