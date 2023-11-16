from contextlib import contextmanager

@contextmanager
def closing(fname):
    if False:
        for i in range(10):
            print('nop')
    f = None
    try:
        f = open(fname, 'r')
        yield f
    finally:
        if f:
            f.close()
with closing('test.txt') as f:
    print(f.read())