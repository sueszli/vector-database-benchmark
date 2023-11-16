import gevent

def f():
    if False:
        for i in range(10):
            print('nop')
    __import__('_imports_at_top_level')
g = gevent.spawn(f)
g.get()