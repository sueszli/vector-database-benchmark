"""Global IPython app to support test running.

We must start our own ipython object and heavily muck with it so that all the
modifications IPython makes to system behavior don't send the doctest machinery
into a fit.  This code should be considered a gross hack, but it gets the job
done.
"""
import builtins as builtin_mod
import sys
import types
from pathlib import Path
from . import tools
from IPython.core import page
from IPython.utils import io
from IPython.terminal.interactiveshell import TerminalInteractiveShell

def get_ipython():
    if False:
        i = 10
        return i + 15
    return start_ipython()

def xsys(self, cmd):
    if False:
        for i in range(10):
            print('nop')
    'Replace the default system call with a capturing one for doctest.\n    '
    print(self.getoutput(cmd, split=False, depth=1).rstrip(), end='', file=sys.stdout)
    sys.stdout.flush()

def _showtraceback(self, etype, evalue, stb):
    if False:
        for i in range(10):
            print('nop')
    'Print the traceback purely on stdout for doctest to capture it.\n    '
    print(self.InteractiveTB.stb2text(stb), file=sys.stdout)

def start_ipython():
    if False:
        for i in range(10):
            print('nop')
    'Start a global IPython shell, which we need for IPython-specific syntax.\n    '
    global get_ipython
    if hasattr(start_ipython, 'already_called'):
        return
    start_ipython.already_called = True
    _displayhook = sys.displayhook
    _excepthook = sys.excepthook
    _main = sys.modules.get('__main__')
    config = tools.default_config()
    config.TerminalInteractiveShell.simple_prompt = True
    shell = TerminalInteractiveShell.instance(config=config)
    shell.tempfiles.append(Path(config.HistoryManager.hist_file))
    shell.builtin_trap.activate()
    shell.system = types.MethodType(xsys, shell)
    shell._showtraceback = types.MethodType(_showtraceback, shell)
    sys.modules['__main__'] = _main
    sys.displayhook = _displayhook
    sys.excepthook = _excepthook
    _ip = shell
    get_ipython = _ip.get_ipython
    builtin_mod._ip = _ip
    builtin_mod.ip = _ip
    builtin_mod.get_ipython = get_ipython

    def nopage(strng, start=0, screen_lines=0, pager_cmd=None):
        if False:
            i = 10
            return i + 15
        if isinstance(strng, dict):
            strng = strng.get('text/plain', '')
        print(strng)
    page.orig_page = page.pager_page
    page.pager_page = nopage
    return _ip