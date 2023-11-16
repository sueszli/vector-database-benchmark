import logging
import threading
try:
    import pythoncom
    HAS_LIBS = True
except ImportError:
    HAS_LIBS = False
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        return 10
    '\n    Only load if required libraries exist\n    '
    if not HAS_LIBS:
        return False
    else:
        return True

class Com:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.need_com_init = not self._is_main_thread()

    def _is_main_thread(self):
        if False:
            for i in range(10):
                print('nop')
        return threading.current_thread().name == 'MainThread'

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        if self.need_com_init:
            log.debug('Initializing COM library')
            pythoncom.CoInitialize()

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            for i in range(10):
                print('nop')
        if self.need_com_init:
            log.debug('Uninitializing COM library')
            pythoncom.CoUninitialize()