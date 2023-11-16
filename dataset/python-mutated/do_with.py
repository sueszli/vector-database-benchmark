from contextlib import contextmanager

@contextmanager
def log(name):
    if False:
        print('Hello World!')
    print('[%s] start...' % name)
    yield
    print('[%s] end.' % name)
with log('DEBUG'):
    print('Hello, world!')
    print('Hello, Python!')