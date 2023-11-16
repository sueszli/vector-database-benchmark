import signal
import logging

class WithKeyboardInterruptAs:

    def __init__(self, callback):
        if False:
            print('Hello World!')
        if callback is None:
            callback = lambda *args, **kwargs: None
        self.callback = callback

    def __enter__(self):
        if False:
            print('Hello World!')
        self.signal_received = 0
        self.old_handler = signal.getsignal(signal.SIGINT)
        try:
            signal.signal(signal.SIGINT, self.handler)
        except ValueError as e:
            logging.debug(e)

    def handler(self, sig, frame):
        if False:
            i = 10
            return i + 15
        self.signal_received += 1
        if self.signal_received > 3:
            self.old_handler(sig, frame)
        else:
            self.callback()
            logging.debug('SIGINT received. Supressing KeyboardInterrupt.')

    def __exit__(self, type, value, traceback):
        if False:
            for i in range(10):
                print('nop')
        try:
            signal.signal(signal.SIGINT, self.old_handler)
        except ValueError as e:
            logging.debug(e)