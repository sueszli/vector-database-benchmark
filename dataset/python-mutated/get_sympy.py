"""Functions to get the correct sympy version to run tests."""
from __future__ import print_function
import os
import sys

def path_hack():
    if False:
        return 10
    '\n    Hack sys.path to import correct (local) sympy.\n    '
    this_file = os.path.abspath(__file__)
    sympy_dir = os.path.join(os.path.dirname(this_file), '..')
    sympy_dir = os.path.normpath(sympy_dir)
    sys.path.insert(0, sympy_dir)
    return sympy_dir