import platform
if platform.system() != 'Windows':
    import signal
else:
    signal = None
__all__ = ['TimeoutError', 'WindowsError', 'TimeoutContext']

class TimeoutError(Exception):
    pass

class WindowsError(Exception):
    pass

class TimeoutContext:
    """Timeout class using ALARM signal."""

    def __init__(self, sec):
        if False:
            for i in range(10):
                print('nop')
        self.sec = sec

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        if signal is None:
            raise WindowsError('Windows is not supported for this test')
        signal.signal(signal.SIGALRM, self.raise_timeout)
        signal.alarm(self.sec)

    def __exit__(self, *args):
        if False:
            while True:
                i = 10
        signal.alarm(0)

    def raise_timeout(self, *args):
        if False:
            i = 10
            return i + 15
        raise TimeoutError('A timeout error have been raised.')