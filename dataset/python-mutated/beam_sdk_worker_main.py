import sys
import pyflink.fn_execution.beam.beam_operations
import pyflink.fn_execution.beam.beam_coders
import apache_beam.runners.worker.sdk_worker_main
from apache_beam.runners.worker import sdk_worker
sdk_worker.DEFAULT_BUNDLE_PROCESSOR_CACHE_SHUTDOWN_THRESHOLD_S = 86400 * 30

def print_to_logging(logging_func, msg, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    if msg != '\n':
        logging_func(msg, *args, **kwargs)

class CustomPrint(object):

    def __init__(self, _print):
        if False:
            for i in range(10):
                print('nop')
        self._msg_buffer = []
        self._print = _print

    def print(self, *args, sep=' ', end='\n', file=None):
        if False:
            for i in range(10):
                print('nop')
        self._msg_buffer.append(sep.join([str(arg) for arg in args]))
        if end == '\n':
            self._print(''.join(self._msg_buffer), sep=sep, end=end, file=file)
            self._msg_buffer.clear()
        else:
            self._msg_buffer.append(end)

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        if self._msg_buffer:
            self._print(''.join(self._msg_buffer), sep='', end='\n')
            self._msg_buffer.clear()

def main():
    if False:
        i = 10
        return i + 15
    import builtins
    import logging
    from functools import partial
    _info = logging.getLogger().info
    _error = logging.getLogger().error
    sys.stdout.write = partial(print_to_logging, _info)
    sys.stderr.write = partial(print_to_logging, _error)
    custom_print = CustomPrint(print)
    builtins.print = custom_print.print
    logging.getLogger().handlers = []
    apache_beam.runners.worker.sdk_worker_main.main(sys.argv)
    custom_print.close()