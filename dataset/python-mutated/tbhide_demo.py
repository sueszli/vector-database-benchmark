import time

def D():
    if False:
        print('Hello World!')
    time.sleep(0.7)

def C():
    if False:
        for i in range(10):
            print('nop')
    __tracebackhide__ = True
    time.sleep(0.1)
    D()

def B():
    if False:
        while True:
            i = 10
    __tracebackhide__ = True
    time.sleep(0.1)
    C()

def A():
    if False:
        for i in range(10):
            print('nop')
    time.sleep(0.1)
    B()
A()