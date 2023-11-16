import shlex
from sys import platform

def on_windows() -> bool:
    if False:
        print('Hello World!')
    return 'win32' in platform

def on_posix() -> bool:
    if False:
        for i in range(10):
            print('nop')
    return not on_windows()

def split_args(args: str) -> list[str]:
    if False:
        return 10
    'Split arguments and add escape characters as appropriate for the OS'
    return shlex.split(args, posix=on_posix())