import os
import time

class MyException(Exception):
    pass

def passing(*args):
    if False:
        i = 10
        return i + 15
    pass

def sleeping(s):
    if False:
        print('Hello World!')
    seconds = s
    while seconds > 0:
        time.sleep(min(seconds, 0.1))
        seconds -= 0.1
    os.environ['ROBOT_THREAD_TESTING'] = str(s)
    return s

def returning(arg):
    if False:
        print('Hello World!')
    return arg

def failing(msg='xxx'):
    if False:
        return 10
    raise MyException(msg)