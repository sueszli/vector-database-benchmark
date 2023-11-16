import gevent

def fn2():
    if False:
        print('Hello World!')
    return 2

def fn():
    if False:
        print('Hello World!')
    return gevent.wait([gevent.spawn(fn2), gevent.spawn(fn2)])
gevent.spawn(fn).get()

def raise_name_error():
    if False:
        while True:
            i = 10
    raise NameError('ThisIsExpected')
try:
    gevent.spawn(raise_name_error).get()
    raise AssertionError('Should fail')
except NameError as e:
    x = e