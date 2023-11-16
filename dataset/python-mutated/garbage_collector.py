import gc
from ..Qt import QtCore

class GarbageCollector(object):
    """
    Disable automatic garbage collection and instead collect manually
    on a timer.

    This is done to ensure that garbage collection only happens in the GUI
    thread, as otherwise Qt can crash.

    Credit:  Erik Janssens
    Source:  http://pydev.blogspot.com/2014/03/should-python-garbage-collector-be.html
    """

    def __init__(self, interval=1.0, debug=False):
        if False:
            i = 10
            return i + 15
        self.debug = debug
        if debug:
            gc.set_debug(gc.DEBUG_LEAK)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.check)
        self.threshold = gc.get_threshold()
        gc.disable()
        self.timer.start(interval * 1000)

    def check(self):
        if False:
            for i in range(10):
                print('nop')
        (l0, l1, l2) = gc.get_count()
        if self.debug:
            print('gc_check called:', l0, l1, l2)
        if l0 > self.threshold[0]:
            num = gc.collect(0)
            if self.debug:
                print('collecting gen 0, found: %d unreachable' % num)
            if l1 > self.threshold[1]:
                num = gc.collect(1)
                if self.debug:
                    print('collecting gen 1, found: %d unreachable' % num)
                if l2 > self.threshold[2]:
                    num = gc.collect(2)
                    if self.debug:
                        print('collecting gen 2, found: %d unreachable' % num)

    def debug_cycles(self):
        if False:
            i = 10
            return i + 15
        gc.collect()
        for obj in gc.garbage:
            print(obj, repr(obj), type(obj))