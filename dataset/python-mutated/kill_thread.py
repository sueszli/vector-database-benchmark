import ctypes
import inspect
import threading
import time
__all__ = ['stop_thread']

def _async_raise(tid, exctype):
    if False:
        i = 10
        return i + 15
    'raises the exception, performs cleanup if needed'
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype):
        exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0:
        raise ValueError('invalid thread id')
    elif res != 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError('PyThreadState_SetAsyncExc failed')

def stop_thread(thread):
    if False:
        print('Hello World!')
    _async_raise(thread.ident, SystemExit)

def test():
    if False:
        i = 10
        return i + 15
    while True:
        print('-------')
        time.sleep(0.5)
if __name__ == '__main__':
    t = threading.Thread(target=test)
    t.start()
    time.sleep(5.2)
    print('main thread sleep finish')
    stop_thread(t)