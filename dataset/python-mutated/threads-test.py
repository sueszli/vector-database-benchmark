import threading
import sys
import numpy as np
try:
    import __builtin__ as builtins
except ImportError:
    import builtins
try:
    builtins.profile
except AttributeError:

    def profile(func):
        if False:
            print('Hello World!')
        return func
    builtins.profile = profile

class MyThread(threading.Thread):

    @profile
    def run(self):
        if False:
            i = 10
            return i + 15
        z = 0
        z = np.random.uniform(0, 100, size=2 * 5000)

class MyThread2(threading.Thread):

    @profile
    def run(self):
        if False:
            return 10
        z = 0
        for i in range(5000 // 2):
            z += 1
use_threads = True
if use_threads:
    for i in range(10000):
        t1 = MyThread()
        t2 = MyThread2()
        t1.start()
        t2.start()
        t1.join()
        t2.join()
else:
    t1 = MyThread()
    t1.run()
    t2 = MyThread2()
    t2.run()