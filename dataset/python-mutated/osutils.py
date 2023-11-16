import sys
from pathlib import Path

def get_home_dir():
    if False:
        while True:
            i = 10
    return Path().home()
if sys.platform == 'win32':

    def get_appstate_dir():
        if False:
            for i in range(10):
                print('nop')
        homedir = get_home_dir()
        winversion = sys.getwindowsversion()
        if winversion[0] == 6:
            appdir = homedir / 'AppData' / 'Roaming' / '.Tribler'
        else:
            appdir = homedir / 'Application Data' / '.Tribler'
        return appdir

    def quote_path_with_spaces(s: str):
        if False:
            return 10
        if s.endswith('.exe'):
            return '"%s"' % s
        return s
else:

    def get_appstate_dir():
        if False:
            return 10
        return get_home_dir() / '.Tribler'

    def quote_path_with_spaces(s: str):
        if False:
            for i in range(10):
                print('nop')
        return s