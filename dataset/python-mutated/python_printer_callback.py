from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import sys
try:
    import IPython
    from IPython.core.interactiveshell import InteractiveShell
    have_ipython = True
except ImportError:
    have_ipython = False

def print_callback(val):
    if False:
        print('Hello World!')
    '\n    Internal function.\n    This function is called via a call back returning from IPC to Cython\n    to Python. It tries to perform incremental printing to IPython Notebook or\n    Jupyter Notebook and when all else fails, just prints locally.\n    '
    success = False
    try:
        if have_ipython:
            if InteractiveShell.initialized():
                IPython.display.publish_display_data({'text/plain': val, 'text/html': '<pre>' + val + '</pre>'})
                success = True
    except:
        pass
    if not success:
        print(val)
        sys.stdout.flush()