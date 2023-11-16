import logging
import queue
import threading
from coalib.processes.communication.LogMessage import LogMessage

class LogPrinterThread(threading.Thread):
    """
    This is the Thread object that outputs all log messages it gets from
    its message_queue. Setting obj.running = False will stop within the next
    0.1 seconds.
    """

    def __init__(self, message_queue, log_printer=None):
        if False:
            print('Hello World!')
        threading.Thread.__init__(self)
        self.running = True
        self.message_queue = message_queue

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        while self.running:
            try:
                elem = self.message_queue.get(timeout=0.1)
                if isinstance(elem, LogMessage):
                    logging.log(elem.log_level, elem.message)
                else:
                    logging.info(elem)
            except queue.Empty:
                pass