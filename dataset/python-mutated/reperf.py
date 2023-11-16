import re
import time

def main():
    if False:
        while True:
            i = 10
    s = '\x0bhello\x0c \x0bworld\x0c ' * 1000
    p = re.compile('([\\13\\14])')
    timefunc(10, p.sub, '', s)
    timefunc(10, p.split, s)
    timefunc(10, p.findall, s)

def timefunc(n, func, *args, **kw):
    if False:
        for i in range(10):
            print('nop')
    t0 = time.perf_counter()
    try:
        for i in range(n):
            result = func(*args, **kw)
        return result
    finally:
        t1 = time.perf_counter()
        if n > 1:
            print(n, 'times', end=' ')
        print(func.__name__, '%.3f' % (t1 - t0), 'CPU seconds')
main()