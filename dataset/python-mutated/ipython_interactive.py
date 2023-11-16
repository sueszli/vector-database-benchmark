"""
Command to start an interactive IPython prompt.
"""
from __future__ import annotations
import sys
from contextlib import contextmanager
import gdb
import pwndbg.color.message as M
import pwndbg.commands
import pwndbg.lib.stdio

@contextmanager
def switch_to_ipython_env():
    if False:
        print('Hello World!')
    "We need to change stdout/stderr to the default ones, otherwise we can't use tab or autocomplete"
    saved_excepthook = sys.excepthook
    with pwndbg.lib.stdio.stdio:
        yield
    sys.ps1 = '>>> '
    sys.ps2 = '... '
    sys.excepthook = saved_excepthook

@pwndbg.commands.ArgparsedCommand('Start an interactive IPython prompt.')
def ipi() -> None:
    if False:
        while True:
            i = 10
    with switch_to_ipython_env():
        try:
            gdb.execute('pi import IPython')
        except gdb.error:
            print(M.warn('Cannot import IPython.\nYou need to install IPython if you want to use this command.\nMaybe you can try `pip install ipython` first.'))
            return
        code4ipython = "import jedi\nimport pwn\njedi.Interpreter._allow_descriptor_getattr_default = False\nIPython.embed(colors='neutral',banner1='',confirm_exit=False,simple_prompt=False, user_ns=globals())\n"
        gdb.execute(f'py\n{code4ipython}')