""" 'editor' hooks for common editors that work well with ipython

They should honor the line number argument, at least.

Contributions are *very* welcome.
"""
import os
import shlex
import subprocess
import sys
from IPython import get_ipython
from IPython.core.error import TryNext
from IPython.utils import py3compat

def install_editor(template, wait=False):
    if False:
        i = 10
        return i + 15
    "Installs the editor that is called by IPython for the %edit magic.\n\n    This overrides the default editor, which is generally set by your EDITOR\n    environment variable or is notepad (windows) or vi (linux). By supplying a\n    template string `run_template`, you can control how the editor is invoked\n    by IPython -- (e.g. the format in which it accepts command line options)\n\n    Parameters\n    ----------\n    template : basestring\n        run_template acts as a template for how your editor is invoked by\n        the shell. It should contain '{filename}', which will be replaced on\n        invocation with the file name, and '{line}', $line by line number\n        (or 0) to invoke the file with.\n    wait : bool\n        If `wait` is true, wait until the user presses enter before returning,\n        to facilitate non-blocking editors that exit immediately after\n        the call.\n    "

    def call_editor(self, filename, line=0):
        if False:
            i = 10
            return i + 15
        if line is None:
            line = 0
        cmd = template.format(filename=shlex.quote(filename), line=line)
        print('>', cmd)
        if sys.platform.startswith('win'):
            cmd = shlex.split(cmd)
        proc = subprocess.Popen(cmd, shell=True)
        if proc.wait() != 0:
            raise TryNext()
        if wait:
            py3compat.input('Press Enter when done editing:')
    get_ipython().set_hook('editor', call_editor)
    get_ipython().editor = template

def komodo(exe=u'komodo'):
    if False:
        for i in range(10):
            print('nop')
    ' Activestate Komodo [Edit] '
    install_editor(exe + u' -l {line} {filename}', wait=True)

def scite(exe=u'scite'):
    if False:
        for i in range(10):
            print('nop')
    ' SciTE or Sc1 '
    install_editor(exe + u' {filename} -goto:{line}')

def notepadplusplus(exe=u'notepad++'):
    if False:
        i = 10
        return i + 15
    ' Notepad++ http://notepad-plus.sourceforge.net '
    install_editor(exe + u' -n{line} {filename}')

def jed(exe=u'jed'):
    if False:
        i = 10
        return i + 15
    ' JED, the lightweight emacsish editor '
    install_editor(exe + u' +{line} {filename}')

def idle(exe=u'idle'):
    if False:
        print('Hello World!')
    ' Idle, the editor bundled with python\n\n    Parameters\n    ----------\n    exe : str, None\n        If none, should be pretty smart about finding the executable.\n    '
    if exe is None:
        import idlelib
        p = os.path.dirname(idlelib.__filename__)
        exe = os.path.join(p, 'idle.py')
    install_editor(exe + u' {filename}')

def mate(exe=u'mate'):
    if False:
        i = 10
        return i + 15
    ' TextMate, the missing editor'
    install_editor(exe + u' -w -l {line} {filename}')

def emacs(exe=u'emacs'):
    if False:
        while True:
            i = 10
    install_editor(exe + u' +{line} {filename}')

def gnuclient(exe=u'gnuclient'):
    if False:
        print('Hello World!')
    install_editor(exe + u' -nw +{line} {filename}')

def crimson_editor(exe=u'cedt.exe'):
    if False:
        return 10
    install_editor(exe + u' /L:{line} {filename}')

def kate(exe=u'kate'):
    if False:
        i = 10
        return i + 15
    install_editor(exe + u' -u -l {line} {filename}')