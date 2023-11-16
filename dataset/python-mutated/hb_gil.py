"""
Run this script in the qtconsole with one of:

    %load hb_gil.py

or
    %run hb_gil.py

Holding the GIL for too long could disrupt the heartbeat.

See Issue #1260: https://github.com/ipython/ipython/issues/1260

"""
import sys
import time
from cython import inline

def gilsleep(t):
    if False:
        i = 10
        return i + 15
    'gil-holding sleep with cython.inline'
    code = '\n'.join(['from posix cimport unistd', 'unistd.sleep(t)'])
    while True:
        inline(code, quiet=True, t=t)
        print(time.time())
        sys.stdout.flush()
gilsleep(5)