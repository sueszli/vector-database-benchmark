import sys
import threading
import time
COUNT = 100

def slow_tracefunc(frame, event, arg):
    if False:
        while True:
            i = 10
    time.sleep(0.01)
    return slow_tracefunc

def run_with_slow_tracefunc(fn):
    if False:
        return 10
    sys.settrace(slow_tracefunc)
    return fn()

def outer():
    if False:
        for i in range(10):
            print('nop')
    x = 0
    done = [False]

    def traced_looper():
        if False:
            return 10
        print(locals())
        nonlocal x
        count = 0
        while not done[0]:
            count += 1
        return count
    t = threading.Thread(target=run_with_slow_tracefunc, args=(traced_looper,))
    t.start()
    for i in range(COUNT):
        print(f'after {i} increments, x is {x}')
        x += 1
        time.sleep(0.01)
    done[0] = True
    t.join()
    print(f'Final discrepancy: {COUNT - x} (should be 0)')
outer()