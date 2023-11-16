from __future__ import print_function
import gevent
from gevent import monkey
monkey.patch_all()
import sys
from multiprocessing import Process
from subprocess import Popen, PIPE
from gevent import testing as greentest

def f(sleep_sec):
    if False:
        return 10
    gevent.sleep(sleep_sec)

class TestIssue600(greentest.TestCase):
    __timeout__ = greentest.LARGE_TIMEOUT

    @greentest.skipOnLibuvOnPyPyOnWin('hangs')
    def test_invoke(self):
        if False:
            i = 10
            return i + 15
        p = Popen([sys.executable, '-V'], stdout=PIPE, stderr=PIPE)
        gevent.sleep(0)
        p.communicate()
        gevent.sleep(0)

    def test_process(self):
        if False:
            print('Hello World!')
        p = Process(target=f, args=(0.5,))
        p.start()
        with gevent.Timeout(3):
            p.join(10)
if __name__ == '__main__':
    greentest.main()