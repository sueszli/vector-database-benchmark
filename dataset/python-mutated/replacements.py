"""Improved replacements for standard functions
"""
import time as __time

def sleep(n):
    if False:
        for i in range(10):
            print('nop')
    'sleep(n)\n\n    Replacement for :func:`time.sleep()`, which does not return if a signal is received.\n\n    Arguments:\n      n (int):  Number of seconds to sleep.\n    '
    end = __time.time() + n
    while True:
        left = end - __time.time()
        if left <= 0:
            break
        __time.sleep(left)