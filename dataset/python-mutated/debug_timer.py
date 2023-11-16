from __future__ import print_function
from __future__ import division
import click
import time

def timeit(method):
    if False:
        i = 10
        return i + 15
    'From: https://www.andreas-jung.com/contents/a-python-decorator-for-measuring-the-execution-time-of-methods  # NOQA\n    '

    def timed(*args, **kw):
        if False:
            while True:
                i = 10
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        message = '%r (%r, %r) %2.2f sec' % (method.__name__, args, kw, te - ts)
        click.secho(message + '\n', fg='red')
        return result
    return timed