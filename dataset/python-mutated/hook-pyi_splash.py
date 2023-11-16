"""
This hook does not move a module that can be installed by a package manager, but points to a PyInstaller internal
module that can be imported into the users python instance.

The module is implemented in 'PyInstaller/fake-modules/pyi_splash.py'.
"""
import os
from PyInstaller import PACKAGEPATH
from PyInstaller.utils.hooks import logger

def pre_find_module_path(api):
    if False:
        for i in range(10):
            print('nop')
    try:
        import pyi_splash
    except ImportError:
        module_dir = os.path.join(PACKAGEPATH, 'fake-modules')
        api.search_dirs = [module_dir]
        logger.info('Adding pyi_splash module to application dependencies.')
    else:
        logger.info('A local module named "pyi_splash" is installed. Use the installed one instead.')
        return