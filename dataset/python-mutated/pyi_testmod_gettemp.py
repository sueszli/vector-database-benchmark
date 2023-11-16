import os
import sys

def gettemp(basename):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the path to a temp file previously written by the temp runner.\n    Useful to compare results between running in interpreter and running frozen.\n    '
    exec_dir = os.path.dirname(sys.executable)
    file_onedir = os.path.join(exec_dir, '..', '..', basename)
    file_onefile = os.path.join(exec_dir, '..', basename)
    if os.path.exists(file_onedir):
        return file_onedir
    else:
        return file_onefile