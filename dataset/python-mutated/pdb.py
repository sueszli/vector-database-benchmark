import multiprocessing
import os
import pdb
import sys
__all__ = ['set_trace']
_stdin = [None]
_stdin_lock = multiprocessing.Lock()
try:
    _stdin_fd = sys.stdin.fileno()
except Exception:
    _stdin_fd = None

class MultiprocessingPdb(pdb.Pdb):
    """A Pdb wrapper that works in a multiprocessing environment.

    Usage: `from fairseq import pdb; pdb.set_trace()`
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        pdb.Pdb.__init__(self, nosigint=True)

    def _cmdloop(self):
        if False:
            i = 10
            return i + 15
        stdin_bak = sys.stdin
        with _stdin_lock:
            try:
                if _stdin_fd is not None:
                    if not _stdin[0]:
                        _stdin[0] = os.fdopen(_stdin_fd)
                    sys.stdin = _stdin[0]
                self.cmdloop()
            finally:
                sys.stdin = stdin_bak

def set_trace():
    if False:
        for i in range(10):
            print('nop')
    pdb = MultiprocessingPdb()
    pdb.set_trace(sys._getframe().f_back)