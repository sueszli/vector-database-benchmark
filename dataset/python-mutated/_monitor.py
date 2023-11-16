import atexit
from threading import Event, Thread, current_thread
from time import time
from warnings import warn
__all__ = ['TMonitor', 'TqdmSynchronisationWarning']

class TqdmSynchronisationWarning(RuntimeWarning):
    """tqdm multi-thread/-process errors which may cause incorrect nesting
    but otherwise no adverse effects"""
    pass

class TMonitor(Thread):
    """
    Monitoring thread for tqdm bars.
    Monitors if tqdm bars are taking too much time to display
    and readjusts miniters automatically if necessary.

    Parameters
    ----------
    tqdm_cls  : class
        tqdm class to use (can be core tqdm or a submodule).
    sleep_interval  : float
        Time to sleep between monitoring checks.
    """
    _test = {}

    def __init__(self, tqdm_cls, sleep_interval):
        if False:
            while True:
                i = 10
        Thread.__init__(self)
        self.daemon = True
        self.woken = 0
        self.tqdm_cls = tqdm_cls
        self.sleep_interval = sleep_interval
        self._time = self._test.get('time', time)
        self.was_killed = self._test.get('Event', Event)()
        atexit.register(self.exit)
        self.start()

    def exit(self):
        if False:
            return 10
        self.was_killed.set()
        if self is not current_thread():
            self.join()
        return self.report()

    def get_instances(self):
        if False:
            print('Hello World!')
        return [i for i in self.tqdm_cls._instances.copy() if hasattr(i, 'start_t')]

    def run(self):
        if False:
            i = 10
            return i + 15
        cur_t = self._time()
        while True:
            self.woken = cur_t
            self.was_killed.wait(self.sleep_interval)
            if self.was_killed.is_set():
                return
            with self.tqdm_cls.get_lock():
                cur_t = self._time()
                instances = self.get_instances()
                for instance in instances:
                    if self.was_killed.is_set():
                        return
                    if instance.miniters > 1 and cur_t - instance.last_print_t >= instance.maxinterval:
                        instance.miniters = 1
                        instance.refresh(nolock=True)
                    del instance
                if instances != self.get_instances():
                    warn('Set changed size during iteration' + ' (see https://github.com/tqdm/tqdm/issues/481)', TqdmSynchronisationWarning, stacklevel=2)
                del instances

    def report(self):
        if False:
            i = 10
            return i + 15
        return not self.was_killed.is_set()