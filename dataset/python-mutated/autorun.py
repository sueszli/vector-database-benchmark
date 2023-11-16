"""
Run commands when the Scapy interpreter starts.
"""
import builtins
import code
from io import StringIO
import logging
from queue import Queue
import sys
import threading
import traceback
from scapy.config import conf
from scapy.themes import NoTheme, DefaultTheme, HTMLTheme2, LatexTheme2
from scapy.error import log_scapy, Scapy_Exception
from scapy.utils import tex_escape
from typing import Any, Optional, TextIO, Dict, Tuple

class StopAutorun(Scapy_Exception):
    code_run = ''

class StopAutorunTimeout(StopAutorun):
    pass

class ScapyAutorunInterpreter(code.InteractiveInterpreter):

    def __init__(self, *args, **kargs):
        if False:
            i = 10
            return i + 15
        code.InteractiveInterpreter.__init__(self, *args, **kargs)

    def write(self, data):
        if False:
            print('Hello World!')
        pass

def autorun_commands(_cmds, my_globals=None, verb=None):
    if False:
        return 10
    sv = conf.verb
    try:
        try:
            if my_globals is None:
                from scapy.main import _scapy_builtins
                my_globals = _scapy_builtins()
            interp = ScapyAutorunInterpreter(locals=my_globals)
            try:
                del builtins.__dict__['scapy_session']['_']
            except KeyError:
                pass
            if verb is not None:
                conf.verb = verb
            cmd = ''
            cmds = _cmds.splitlines()
            cmds.append('')
            cmds.reverse()
            while True:
                if cmd:
                    sys.stderr.write(sys.__dict__.get('ps2', '... '))
                else:
                    sys.stderr.write(sys.__dict__.get('ps1', '>>> '))
                line = cmds.pop()
                print(line)
                cmd += '\n' + line
                sys.last_value = None
                if interp.runsource(cmd):
                    continue
                if sys.last_value:
                    traceback.print_exception(sys.last_type, sys.last_value, sys.last_traceback.tb_next, file=sys.stdout)
                    sys.last_value = None
                    return False
                cmd = ''
                if len(cmds) <= 1:
                    break
        except SystemExit:
            pass
    finally:
        conf.verb = sv
    try:
        return builtins.__dict__['scapy_session']['_']
    except KeyError:
        return builtins.__dict__.get('_', None)

def autorun_commands_timeout(cmds, timeout=None, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Wraps autorun_commands with a timeout that raises StopAutorunTimeout\n    on expiration.\n    '
    if timeout is None:
        return autorun_commands(cmds, **kwargs)
    q = Queue()

    def _runner():
        if False:
            print('Hello World!')
        q.put(autorun_commands(cmds, **kwargs))
    th = threading.Thread(target=_runner)
    th.daemon = True
    th.start()
    th.join(timeout)
    if th.is_alive():
        raise StopAutorunTimeout
    return q.get()

class StringWriter(StringIO):
    """Util to mock sys.stdout and sys.stderr, and
    store their output in a 's' var."""

    def __init__(self, debug=None):
        if False:
            while True:
                i = 10
        self.s = ''
        self.debug = debug
        super().__init__()

    def write(self, x):
        if False:
            i = 10
            return i + 15
        if getattr(self, 'debug', None) and self.debug:
            self.debug.write(x)
        if getattr(self, 's', None) is not None:
            self.s += x
        return len(x)

    def flush(self):
        if False:
            while True:
                i = 10
        if getattr(self, 'debug', None) and self.debug:
            self.debug.flush()

def autorun_get_interactive_session(cmds, **kargs):
    if False:
        i = 10
        return i + 15
    'Create an interactive session and execute the\n    commands passed as "cmds" and return all output\n\n    :param cmds: a list of commands to run\n    :param timeout: timeout in seconds\n    :returns: (output, returned) contains both sys.stdout and sys.stderr logs\n    '
    (sstdout, sstderr, sexcepthook) = (sys.stdout, sys.stderr, sys.excepthook)
    sw = StringWriter()
    h_old = log_scapy.handlers[0]
    log_scapy.removeHandler(h_old)
    log_scapy.addHandler(logging.StreamHandler(stream=sw))
    try:
        try:
            sys.stdout = sys.stderr = sw
            sys.excepthook = sys.__excepthook__
            res = autorun_commands_timeout(cmds, **kargs)
        except StopAutorun as e:
            e.code_run = sw.s
            raise
    finally:
        (sys.stdout, sys.stderr, sys.excepthook) = (sstdout, sstderr, sexcepthook)
        log_scapy.removeHandler(log_scapy.handlers[0])
        log_scapy.addHandler(h_old)
    return (sw.s, res)

def autorun_get_interactive_live_session(cmds, **kargs):
    if False:
        while True:
            i = 10
    'Create an interactive session and execute the\n    commands passed as "cmds" and return all output\n\n    :param cmds: a list of commands to run\n    :param timeout: timeout in seconds\n    :returns: (output, returned) contains both sys.stdout and sys.stderr logs\n    '
    (sstdout, sstderr) = (sys.stdout, sys.stderr)
    sw = StringWriter(debug=sstdout)
    try:
        try:
            sys.stdout = sys.stderr = sw
            res = autorun_commands_timeout(cmds, **kargs)
        except StopAutorun as e:
            e.code_run = sw.s
            raise
    finally:
        (sys.stdout, sys.stderr) = (sstdout, sstderr)
    return (sw.s, res)

def autorun_get_text_interactive_session(cmds, **kargs):
    if False:
        while True:
            i = 10
    ct = conf.color_theme
    try:
        conf.color_theme = NoTheme()
        (s, res) = autorun_get_interactive_session(cmds, **kargs)
    finally:
        conf.color_theme = ct
    return (s, res)

def autorun_get_live_interactive_session(cmds, **kargs):
    if False:
        i = 10
        return i + 15
    ct = conf.color_theme
    try:
        conf.color_theme = DefaultTheme()
        (s, res) = autorun_get_interactive_live_session(cmds, **kargs)
    finally:
        conf.color_theme = ct
    return (s, res)

def autorun_get_ansi_interactive_session(cmds, **kargs):
    if False:
        print('Hello World!')
    ct = conf.color_theme
    try:
        conf.color_theme = DefaultTheme()
        (s, res) = autorun_get_interactive_session(cmds, **kargs)
    finally:
        conf.color_theme = ct
    return (s, res)

def autorun_get_html_interactive_session(cmds, **kargs):
    if False:
        return 10
    ct = conf.color_theme

    def to_html(s):
        if False:
            return 10
        return s.replace('<', '&lt;').replace('>', '&gt;').replace('#[#', '<').replace('#]#', '>')
    try:
        try:
            conf.color_theme = HTMLTheme2()
            (s, res) = autorun_get_interactive_session(cmds, **kargs)
        except StopAutorun as e:
            e.code_run = to_html(e.code_run)
            raise
    finally:
        conf.color_theme = ct
    return (to_html(s), res)

def autorun_get_latex_interactive_session(cmds, **kargs):
    if False:
        for i in range(10):
            print('nop')
    ct = conf.color_theme

    def to_latex(s):
        if False:
            i = 10
            return i + 15
        return tex_escape(s).replace('@[@', '{').replace('@]@', '}').replace('@`@', '\\')
    try:
        try:
            conf.color_theme = LatexTheme2()
            (s, res) = autorun_get_interactive_session(cmds, **kargs)
        except StopAutorun as e:
            e.code_run = to_latex(e.code_run)
            raise
    finally:
        conf.color_theme = ct
    return (to_latex(s), res)