import logging
from .runtime_base import RuntimeBase
__all__ = []

class CollectiveRuntime(RuntimeBase):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()

    def _init_worker(self):
        if False:
            i = 10
            return i + 15
        logging.warn("You should not call 'init_worker' method for collective mode.")

    def _run_worker(self):
        if False:
            print('Hello World!')
        logging.warn("You should not call 'run_worker' method for collective mode.")

    def _init_server(self, *args, **kwargs):
        if False:
            print('Hello World!')
        logging.warn("You should not call 'init_server' method for collective mode.")

    def _run_server(self):
        if False:
            while True:
                i = 10
        logging.warn("You should not call 'run_server' method for collective mode.")

    def _stop_worker(self):
        if False:
            i = 10
            return i + 15
        logging.warn("You should not call 'stop_worker' method for collective mode.")