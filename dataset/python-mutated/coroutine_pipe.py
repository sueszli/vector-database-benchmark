"""
Inserting `tqdm` as a "pipe" in a chain of coroutines.
Not to be confused with `asyncio.coroutine`.
"""
from functools import wraps
from tqdm.auto import tqdm

def autonext(func):
    if False:
        print('Hello World!')

    @wraps(func)
    def inner(*args, **kwargs):
        if False:
            while True:
                i = 10
        res = func(*args, **kwargs)
        next(res)
        return res
    return inner

@autonext
def tqdm_pipe(target, **tqdm_kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Coroutine chain pipe `send()`ing to `target`.\n\n    This:\n    >>> r = receiver()\n    >>> p = producer(r)\n    >>> next(r)\n    >>> next(p)\n\n    Becomes:\n    >>> r = receiver()\n    >>> t = tqdm.pipe(r)\n    >>> p = producer(t)\n    >>> next(r)\n    >>> next(p)\n    '
    with tqdm(**tqdm_kwargs) as pbar:
        while True:
            obj = (yield)
            target.send(obj)
            pbar.update()

def source(target):
    if False:
        return 10
    for i in ['foo', 'bar', 'baz', 'pythonista', 'python', 'py']:
        target.send(i)
    target.close()

@autonext
def grep(pattern, target):
    if False:
        while True:
            i = 10
    while True:
        line = (yield)
        if pattern in line:
            target.send(line)

@autonext
def sink():
    if False:
        print('Hello World!')
    while True:
        line = (yield)
        tqdm.write(line)
if __name__ == '__main__':
    source(tqdm_pipe(grep('python', sink())))