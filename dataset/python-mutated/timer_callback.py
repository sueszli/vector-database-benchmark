import pyb
from pyb import Timer

def cb1(t):
    if False:
        print('Hello World!')
    print('cb1')
    t.callback(None)

def cb2(t):
    if False:
        for i in range(10):
            print('nop')
    print('cb2')
    t.deinit()

def cb3(x):
    if False:
        for i in range(10):
            print('nop')
    y = x

    def cb4(t):
        if False:
            while True:
                i = 10
        print('cb4', y)
        t.callback(None)
    return cb4
tim = Timer(1, freq=100, callback=cb1)
pyb.delay(5)
print('before cb1')
pyb.delay(15)
tim = Timer(2, freq=100, callback=cb2)
pyb.delay(5)
print('before cb2')
pyb.delay(15)
tim = Timer(4)
tim.init(freq=100)
tim.callback(cb1)
pyb.delay(5)
print('before cb1')
pyb.delay(15)
tim.init(freq=100)
tim.callback(cb3(3))
pyb.delay(5)
print('before cb4')
pyb.delay(15)