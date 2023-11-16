__author__ = 'Cyril Jaquier'
__copyright__ = 'Copyright (c) 2004 Cyril Jaquier'
__license__ = 'GPL'
import sys
from threading import Thread
from abc import abstractmethod
from .utils import Utils
from ..helpers import excepthook, prctl_set_th_name

class JailThread(Thread):
    """Abstract class for threading elements in Fail2Ban.

	Attributes
	----------
	daemon
	ident
	name
	status
	active : bool
		Control the state of the thread.
	idle : bool
		Control the idle state of the thread.
	sleeptime : int
		The time the thread sleeps for in the loop.
	"""

    def __init__(self, name=None):
        if False:
            while True:
                i = 10
        super(JailThread, self).__init__(name=name)
        self.daemon = True
        self.active = None
        self.idle = False
        self.sleeptime = Utils.DEFAULT_SLEEP_TIME
        run = self.run

        def run_with_except_hook(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            try:
                run(*args, **kwargs)
                self.onStop()
            except Exception as e:
                if sys is not None:
                    excepthook(*sys.exc_info())
                else:
                    print(e)
        self.run = run_with_except_hook

    def _bootstrap(self):
        if False:
            return 10
        prctl_set_th_name(self.name)
        return super(JailThread, self)._bootstrap()

    @abstractmethod
    def status(self, flavor='basic'):
        if False:
            print('Hello World!')
        'Abstract - Should provide status information.\n\t\t'
        pass

    def start(self):
        if False:
            return 10
        'Sets active flag and starts thread.\n\t\t'
        self.active = True
        super(JailThread, self).start()

    @abstractmethod
    def onStop(self):
        if False:
            while True:
                i = 10
        'Abstract - Called when thread ends (after run).\n\t\t'
        pass

    def stop(self):
        if False:
            return 10
        'Sets `active` property to False, to flag run method to return.\n\t\t'
        self.active = False

    @abstractmethod
    def run(self):
        if False:
            for i in range(10):
                print('nop')
        'Abstract - Called when thread starts, thread stops when returns.\n\t\t'
        pass

    def join(self):
        if False:
            for i in range(10):
                print('nop')
        ' Safer join, that could be called also for not started (or ended) threads (used for cleanup).\n\t\t'
        if self.active is not None:
            super(JailThread, self).join()
if not hasattr(JailThread, 'isAlive'):
    JailThread.isAlive = JailThread.is_alive