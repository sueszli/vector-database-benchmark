import time

def f():
    if False:
        print('Hello World!')
    [time.sleep(1) for _ in range(1000)]
f()