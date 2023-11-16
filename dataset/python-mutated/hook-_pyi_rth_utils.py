"""
This hook allows discovery and collection of PyInstaller's internal _pyi_rth_utils module that provides utility
functions for run-time hooks.

The module is implemented in 'PyInstaller/fake-modules/_pyi_rth_utils.py'.
"""
import os
from PyInstaller import PACKAGEPATH

def pre_find_module_path(api):
    if False:
        i = 10
        return i + 15
    module_dir = os.path.join(PACKAGEPATH, 'fake-modules')
    api.search_dirs = [module_dir]