"""
This module defines a utility to write tests (for e.g. flexx.event)
that run both in Python and JS.

This is part of flexx.event and not in flexx/event/tests because of (re)imports
during tests.
"""
import sys
from pscript.functions import py2js, evaljs
from pscript.stdlib import get_std_info, get_partial_std_lib
from ._loop import loop, this_is_js
from ._component import Component
from ._property import Property
from ._js import create_js_component_class, JS_EVENT

class StdoutMismatchError(Exception):
    """ Raised when the stdout mismatches.
    """
    pass

class FakeStream:
    """ To capture Pythons stdout and stderr during the both-tests.
    """

    def __init__(self):
        if False:
            return 10
        self._parts = []

    def write(self, msg):
        if False:
            while True:
                i = 10
        msg2 = msg.replace('\n', '\r')
        if msg.endswith('\n'):
            self._parts.append(msg2[:-1] + '\n')
        else:
            self._parts.append(msg2)

    def flush(self):
        if False:
            while True:
                i = 10
        pass

    def getvalue(self):
        if False:
            i = 10
            return i + 15
        return ''.join(self._parts)

def call_func_in_py(func):
    if False:
        i = 10
        return i + 15
    " Call a function and capture it's stdout.\n    "
    loop.integrate(reset=True)
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    fake_stdout = FakeStream()
    sys.stdout = sys.stderr = fake_stdout
    try:
        func()
    finally:
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
    loop.reset()
    return fake_stdout.getvalue().rstrip()

def call_func_in_js(func, classes, extra_nodejs_args=None):
    if False:
        while True:
            i = 10
    all_classes = []
    for cls in classes:
        for c in cls.mro():
            if c is Component or c is Property or c in all_classes:
                break
            all_classes.append(c)
    code = JS_EVENT
    for c in reversed(all_classes):
        code += create_js_component_class(c, c.__name__, c.__bases__[0].__name__ + '.prototype')
    code += py2js(func, 'test', inline_stdlib=False, docstrings=False)
    code += 'test();loop.reset();'
    (nargs, function_deps, method_deps) = get_std_info(code)
    code = get_partial_std_lib(function_deps, method_deps, []) + code
    return evaljs(code, print_result=False, extra_nodejs_args=extra_nodejs_args)

def smart_compare(func, *comparations):
    if False:
        return 10
    ' Compare multiple text-pairs, raising an error that shows where\n    the texts differ for each of the mismatching pairs.\n    Each comparison should be (name, text, reference).\n    '
    err_msgs = []
    has_errors = False
    for comp in comparations:
        err_msg = validate_text(*comp)
        if err_msg:
            has_errors = True
            err_msgs.append(err_msg)
        else:
            err_msgs.append(' ' * 8 + comp[0] + ' matches the reference\n')
    if has_errors:
        j = '_' * 79 + '\n'
        err_msgs = [''] + err_msgs + ['']
        t = 'Text mismatch in\nFile "%s", line %i, in %s:\n%s'
        raise StdoutMismatchError(t % (func.__code__.co_filename, func.__code__.co_firstlineno, func.__name__, j.join(err_msgs)))

def validate_text(name, text, reference):
    if False:
        return 10
    ' Compare text with a reference. Returns None if they match, and otherwise\n    an error message that outlines where they differ.\n    '
    lines1 = text.split('\n')
    lines2 = reference.split('\n')
    n = max(len(lines1), len(lines2))
    for i in range(len(lines1)):
        if lines1[i].startswith(('[E ', '[W ', '[I ')):
            lines1[i] = lines1[i].split(']', 1)[-1].lstrip()
    while len(lines1) < n:
        lines1.append('')
    while len(lines2) < n:
        lines2.append('')
    nchars = 35
    for i in range(n):
        (line1, line2) = (lines1[i], lines2[i])
        line1 = line1.lower()
        line2 = line2.lower()
        if line2.startswith('?'):
            equal_enough = line2[1:].strip() in line1
        else:
            equal_enough = line1 == line2
        if not equal_enough:
            i1 = max(0, i - 16)
            i2 = min(n, i + 16)
            msg = ' ' * 8 + name.ljust(nchars) + ' ' + 'Reference'.ljust(nchars) + '\n'
            for j in range(i1, i2):
                linenr = str(j + 1).rjust(3, '0')
                prefix = ' >> ' if j == i else '    '
                msg += '{}{} '.format(prefix, linenr)
                msg += _zip(_wrap(lines1[j], nchars, 3), _wrap(lines2[j], nchars, 3), 8)
            return msg

def _wrap(line, nchars, maxlines):
    if False:
        while True:
            i = 10
    line = line.replace('\n', '\\n').replace('\r', '\\r')
    lines = []
    while line:
        lines.append(line[:nchars])
        line = line[nchars:].lstrip()
    if not lines:
        lines.append('·' * nchars)
    elif len(lines) == 1:
        lines[-1] = lines[-1].ljust(nchars, '·')
    elif len(lines) <= maxlines:
        lines[-1] = lines[-1].ljust(nchars, ' ')
    else:
        lines = lines[:maxlines]
        lines[-1] = lines[-1][:-1] + '…'
    return lines

def _zip(lines1, lines2, offset):
    if False:
        i = 10
        return i + 15
    n = max(len(lines1), len(lines2))
    nchars = len(lines1[0])
    while len(lines1) < n:
        lines1.append(' ' * nchars)
    while len(lines2) < n:
        lines2.append(' ' * nchars)
    text = ''
    i = 0
    for (line1, line2) in zip(lines1, lines2):
        if i > 0:
            text += ' ' * offset
        i += 1
        text += line1 + ' ' + line2 + '\n'
    return text

def run_in_both(*classes, js=True, py=True, extra_nodejs_args=None):
    if False:
        print('Hello World!')
    ' Decorator to run a test in both Python and JS.\n\n    The decorator should be provided with any Component classes that\n    you want to use in the test.\n\n    The function docstring should match the stdout + stderr of the test (case\n    insensitive). To provide separate reference outputs for Python and\n    JavaScript, use a delimiter of at least 10 \'-\' characters. Use "? xx"\n    to test that "xx" is present on a line (useful for logged exceptions).\n    '

    def wrapper(func):
        if False:
            while True:
                i = 10
        reference = '\n'.join((line[4:] for line in func.__doc__.splitlines()))
        parts = reference.split('-' * 10)
        pyref = parts[0].strip(' \n')
        jsref = parts[-1].strip(' \n-')

        def runner1():
            if False:
                i = 10
                return i + 15
            err = None
            try:
                return runner2()
            except Exception as e:
                err = e
            if isinstance(err, StdoutMismatchError):
                raise StdoutMismatchError(err)
            elif isinstance(err, RuntimeError):
                raise RuntimeError(err)
            else:
                raise err

        def runner2():
            if False:
                for i in range(10):
                    print('nop')
            if py:
                pyresult = call_func_in_py(func)
                pyresult = pyresult.replace('"', "'").replace("\\'", "'")
                pyresult = pyresult.split('!!!!')[-1]
                pyresult = pyresult.split('old pending sessions\n')[-1]
            if js:
                jsresult = call_func_in_js(func, classes, extra_nodejs_args)
                jsresult = jsresult.replace('\n]', ']').replace('[\n', '[')
                jsresult = jsresult.replace('[  ', '[').replace('  ]', ']')
                jsresult = jsresult.replace('[ ', '[').replace(' ]', ']')
                jsresult = jsresult.replace('\n  ', ' ')
                jsresult = jsresult.replace(',   ', ', ').replace(',  ', ', ')
                jsresult = jsresult.replace('\n}', '}')
                jsresult = jsresult.replace('"', "'").split('!!!!')[-1]
                jsresult = jsresult.replace('null', 'None')
            args = [func]
            if py:
                args.append(('Python', pyresult, pyref))
            if js:
                args.append(('JavaScript', jsresult, jsref))
            smart_compare(*args)
            print(func.__name__, 'ok')
            return True
        return runner1
    return wrapper