"""General console printing utilities used by the Cloud SDK."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import signal
import subprocess
import sys
from fire.console import console_attr
from fire.console import console_pager
from fire.console import encoding
from fire.console import files

def IsInteractive(output=False, error=False, heuristic=False):
    if False:
        return 10
    'Determines if the current terminal session is interactive.\n\n  sys.stdin must be a terminal input stream.\n\n  Args:\n    output: If True then sys.stdout must also be a terminal output stream.\n    error: If True then sys.stderr must also be a terminal output stream.\n    heuristic: If True then we also do some additional heuristics to check if\n               we are in an interactive context. Checking home path for example.\n\n  Returns:\n    True if the current terminal session is interactive.\n  '
    if not sys.stdin.isatty():
        return False
    if output and (not sys.stdout.isatty()):
        return False
    if error and (not sys.stderr.isatty()):
        return False
    if heuristic:
        home = os.getenv('HOME')
        homepath = os.getenv('HOMEPATH')
        if not homepath and (not home or home == '/'):
            return False
    return True

def More(contents, out, prompt=None, check_pager=True):
    if False:
        print('Hello World!')
    'Run a user specified pager or fall back to the internal pager.\n\n  Args:\n    contents: The entire contents of the text lines to page.\n    out: The output stream.\n    prompt: The page break prompt.\n    check_pager: Checks the PAGER env var and uses it if True.\n  '
    if not IsInteractive(output=True):
        out.write(contents)
        return
    if check_pager:
        pager = encoding.GetEncodedValue(os.environ, 'PAGER', None)
        if pager == '-':
            pager = None
        elif not pager:
            for command in ('less', 'pager'):
                if files.FindExecutableOnPath(command):
                    pager = command
                    break
        if pager:
            less_orig = encoding.GetEncodedValue(os.environ, 'LESS', None)
            less = '-R' + (less_orig or '')
            encoding.SetEncodedValue(os.environ, 'LESS', less)
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            p = subprocess.Popen(pager, stdin=subprocess.PIPE, shell=True)
            enc = console_attr.GetConsoleAttr().GetEncoding()
            p.communicate(input=contents.encode(enc))
            p.wait()
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            if less_orig is None:
                encoding.SetEncodedValue(os.environ, 'LESS', None)
            return
    console_pager.Pager(contents, out, prompt).Run()