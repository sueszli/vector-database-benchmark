""" Time taking.

Mostly for measurements of Nuitka of itself, e.g. how long did it take to
call an external tool.
"""
from timeit import default_timer as timer
from nuitka.Tracing import general

class StopWatch(object):
    __slots__ = ('start_time', 'end_time')

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.start_time = None
        self.end_time = None

    def start(self):
        if False:
            print('Hello World!')
        self.start_time = timer()

    def restart(self):
        if False:
            while True:
                i = 10
        self.start()

    def end(self):
        if False:
            i = 10
            return i + 15
        self.end_time = timer()
    stop = end

    def getDelta(self):
        if False:
            for i in range(10):
                print('nop')
        if self.end_time is not None:
            return self.end_time - self.start_time
        else:
            return timer() - self.start_time

class TimerReport(object):
    """Timer that reports how long things took.

    Mostly intended as a wrapper for external process calls.
    """
    __slots__ = ('message', 'decider', 'logger', 'timer', 'min_report_time')

    def __init__(self, message, logger=None, decider=True, min_report_time=None):
        if False:
            for i in range(10):
                print('nop')
        self.message = message
        if decider is True:
            decider = lambda : True
        elif decider is False:
            decider = lambda : False
        if logger is None:
            logger = general
        self.logger = logger
        self.decider = decider
        self.min_report_time = min_report_time
        self.timer = None

    def getTimer(self):
        if False:
            print('Hello World!')
        return self.timer

    def __enter__(self):
        if False:
            while True:
                i = 10
        self.timer = StopWatch()
        self.timer.start()
        return self.timer

    def __exit__(self, exception_type, exception_value, exception_tb):
        if False:
            return 10
        self.timer.end()
        delta_time = self.timer.getDelta()
        above_threshold = self.min_report_time is None or delta_time >= self.min_report_time
        if exception_type is None and above_threshold and self.decider():
            self.logger.info(self.message % self.timer.getDelta())