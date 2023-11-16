"""
`allennlp.common.tqdm.Tqdm` wraps tqdm so we can add configurable
global defaults for certain tqdm parameters.
"""
import logging
from allennlp.common import logging as common_logging
import sys
from time import time
from typing import Optional
try:
    SHELL = str(type(get_ipython()))
except:
    SHELL = ''
if 'zmqshell.ZMQInteractiveShell' in SHELL:
    from tqdm import tqdm_notebook as _tqdm
else:
    from tqdm import tqdm as _tqdm
_tqdm.monitor_interval = 0
logger = logging.getLogger('tqdm')
logger.propagate = False

def replace_cr_with_newline(message: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    "\n    TQDM and requests use carriage returns to get the training line to update for each batch\n    without adding more lines to the terminal output. Displaying those in a file won't work\n    correctly, so we'll just make sure that each batch shows up on its one line.\n    "
    message = message.replace('\r', '').replace('\n', '').replace('\x1b[A', '')
    if message and message[-1] != '\n':
        message += '\n'
    return message

class TqdmToLogsWriter(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.last_message_written_time = 0.0

    def write(self, message):
        if False:
            i = 10
            return i + 15
        file_friendly_message: Optional[str] = None
        if common_logging.FILE_FRIENDLY_LOGGING:
            file_friendly_message = replace_cr_with_newline(message)
            if file_friendly_message.strip():
                sys.stderr.write(file_friendly_message)
        else:
            sys.stderr.write(message)
        now = time()
        if now - self.last_message_written_time >= 10 or '100%' in message:
            if file_friendly_message is None:
                file_friendly_message = replace_cr_with_newline(message)
            for message in file_friendly_message.split('\n'):
                message = message.strip()
                if len(message) > 0:
                    logger.info(message)
                    self.last_message_written_time = now

    def flush(self):
        if False:
            print('Hello World!')
        sys.stderr.flush()

class Tqdm:

    @staticmethod
    def tqdm(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        default_mininterval = 2.0 if common_logging.FILE_FRIENDLY_LOGGING else 0.1
        new_kwargs = {'file': TqdmToLogsWriter(), 'mininterval': default_mininterval, **kwargs}
        return _tqdm(*args, **new_kwargs)

    @staticmethod
    def set_lock(lock):
        if False:
            while True:
                i = 10
        _tqdm.set_lock(lock)

    @staticmethod
    def get_lock():
        if False:
            return 10
        return _tqdm.get_lock()