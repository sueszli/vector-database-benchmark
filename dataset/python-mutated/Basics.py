""" Basics for Nuitka tools.

"""
import os
import sys

def goHome():
    if False:
        return 10
    'Go its own directory, to have it easy with path knowledge.'
    os.chdir(getHomePath())
my_abs_path = os.path.abspath(__file__)

def getHomePath():
    if False:
        i = 10
        return i + 15
    return os.path.normpath(os.path.join(os.path.dirname(my_abs_path), '..', '..'))

def setupPATH():
    if False:
        i = 10
        return i + 15
    'Make sure installed tools are in PATH.\n\n    For Windows, add this to the PATH, so pip installed PyLint will be found\n    near the Python executing this script.\n    '
    os.environ['PATH'] = os.environ['PATH'] + os.pathsep + os.path.join(os.path.dirname(sys.executable), 'scripts')

def addPYTHONPATH(path):
    if False:
        print('Hello World!')
    python_path = os.environ.get('PYTHONPATH', '')
    os.environ['PYTHONPATH'] = os.pathsep.join(python_path.split(os.pathsep) + [path])