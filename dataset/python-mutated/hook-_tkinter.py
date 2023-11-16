import sys
from PyInstaller import compat
from PyInstaller.utils.hooks import logger
from PyInstaller.utils.hooks.tcl_tk import collect_tcl_tk_files

def hook(hook_api):
    if False:
        for i in range(10):
            print('nop')
    '\n    Freeze all external Tcl/Tk data files if this is a supported platform *or* log a non-fatal error otherwise.\n    '
    if compat.is_win or compat.is_darwin or compat.is_unix:
        hook_api.add_datas(collect_tcl_tk_files(hook_api.__file__))
    else:
        logger.error('... skipping Tcl/Tk handling on unsupported platform %s', sys.platform)