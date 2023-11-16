import subprocess
import sys

def getpywin():
    if False:
        i = 10
        return i + 15
    try:
        import win32con
    except ImportError:
        print('pyWin32 not installed but is required...\nInstalling via pip:')
        subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', '--upgrade', 'pip'])
        subprocess.check_call([sys.executable, '-m', 'pip', '-q', 'install', 'pywin32'])
if __name__ == '__main__':
    getpywin()