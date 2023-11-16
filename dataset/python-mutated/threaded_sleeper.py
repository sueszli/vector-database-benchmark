import os
import sys
import time
import threading

def do_sleep():
    if False:
        i = 10
        return i + 15
    while True:
        time.sleep(0.1)
        target = time.time() + 0.1
        while time.time() < target:
            pass

def sleep_a():
    if False:
        i = 10
        return i + 15
    do_sleep()

def sleep_b():
    if False:
        while True:
            i = 10
    do_sleep()

def main():
    if False:
        i = 10
        return i + 15
    sys.stdout.write('%d\n' % (os.getpid(),))
    sys.stdout.flush()
    thread_a = threading.Thread(target=sleep_a)
    thread_a.start()
    thread_b = threading.Thread(target=sleep_b)
    thread_b.start()
if __name__ == '__main__':
    main()