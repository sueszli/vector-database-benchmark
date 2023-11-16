try:
    y = 6 + '7'
except TypeError:
    raise ValueError()
try:
    x = 1 / 0
except ZeroDivisionError:
    raise
raise TypeError()
raise TypeError()
raise TypeError()
raise TypeError()
raise TypeError()
raise TypeError()
raise TypeError()
raise AssertionError
raise AttributeError('test message')

def return_error():
    if False:
        i = 10
        return i + 15
    return ValueError('Something')
raise return_error()

class Class:

    @staticmethod
    def error():
        if False:
            for i in range(10):
                print('nop')
        return ValueError('Something')
raise Class.error()
import ctypes
raise ctypes.WinError(1)
raise IndexError() from ZeroDivisionError
raise IndexError() from ZeroDivisionError
raise IndexError() from ZeroDivisionError
raise IndexError()
raise Foo()