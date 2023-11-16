import os
import sys
from contextlib import contextmanager
from click._compat import isatty
WIN = sys.platform.startswith('win')
CI = 'CI' in os.environ
env = os.environ

@contextmanager
def raw_mode():
    if False:
        return 10
    '\n    Enables terminal raw mode during the context.\n\n    Note: Currently noop for Windows systems.\n\n    Usage: ::\n\n        with raw_mode():\n            do_some_stuff()\n    '
    if WIN or CI:
        yield
    else:
        import tty
        import termios
        if not isatty(sys.stdin):
            f = open('/dev/tty')
            fd = f.fileno()
        else:
            fd = sys.stdin.fileno()
            f = None
        try:
            old_settings = termios.tcgetattr(fd)
            tty.setraw(fd)
        except termios.error:
            pass
        try:
            yield
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                if f is not None:
                    f.close()
            except termios.error:
                pass

def get_default_shell():
    if False:
        for i in range(10):
            print('nop')
    return env.get('DOITLIVE_INTERPRETER') or env.get('SHELL') or '/bin/bash'