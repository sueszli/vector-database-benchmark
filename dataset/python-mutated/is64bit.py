"""is64bit.Python() --> boolean value of detected Python word size. is64bit.os() --> os build version"""
import sys

def Python():
    if False:
        for i in range(10):
            print('nop')
    if sys.platform == 'cli':
        import System
        return System.IntPtr.Size == 8
    else:
        try:
            return sys.maxsize > 2147483647
        except AttributeError:
            return sys.maxint > 2147483647

def os():
    if False:
        while True:
            i = 10
    import platform
    pm = platform.machine()
    if pm != '..' and pm.endswith('64'):
        return True
    else:
        import os
        if 'PROCESSOR_ARCHITEW6432' in os.environ:
            return True
        try:
            return os.environ['PROCESSOR_ARCHITECTURE'].endswith('64')
        except (IndexError, KeyError):
            pass
        try:
            return '64' in platform.architecture()[0]
        except:
            return False
if __name__ == '__main__':
    print('is64bit.Python() =', Python(), 'is64bit.os() =', os())