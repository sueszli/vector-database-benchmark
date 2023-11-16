from __future__ import print_function
import ast
import inspect
import pprint
import sys
import warnings
from datetime import datetime
import functools
from contextlib import contextmanager
from os.path import basename, realpath
from textwrap import dedent
import colorama
import executing
from pygments import highlight
from pygments.formatters import Terminal256Formatter
from pygments.lexers import PythonLexer as PyLexer, Python3Lexer as Py3Lexer
from .coloring import SolarizedDark
PYTHON2 = sys.version_info[0] == 2
_absent = object()
_arg_source_missing = object()

def bindStaticVariable(name, value):
    if False:
        return 10

    def decorator(fn):
        if False:
            return 10
        setattr(fn, name, value)
        return fn
    return decorator

@bindStaticVariable('formatter', Terminal256Formatter(style=SolarizedDark))
@bindStaticVariable('lexer', PyLexer(ensurenl=False) if PYTHON2 else Py3Lexer(ensurenl=False))
def colorize(s):
    if False:
        while True:
            i = 10
    self = colorize
    return highlight(s, self.lexer, self.formatter)

@contextmanager
def supportTerminalColorsInWindows():
    if False:
        for i in range(10):
            print('nop')
    colorama.init()
    yield
    colorama.deinit()

def stderrPrint(*args):
    if False:
        print('Hello World!')
    print(*args, file=sys.stderr)

def isLiteral(s):
    if False:
        print('Hello World!')
    try:
        ast.literal_eval(s)
    except Exception:
        return False
    return True

def colorizedStderrPrint(s):
    if False:
        return 10
    colored = colorize(s)
    with supportTerminalColorsInWindows():
        stderrPrint(colored)
DEFAULT_PREFIX = 'ic| '
DEFAULT_LINE_WRAP_WIDTH = 70
DEFAULT_CONTEXT_DELIMITER = '- '
DEFAULT_OUTPUT_FUNCTION = colorizedStderrPrint
DEFAULT_ARG_TO_STRING_FUNCTION = pprint.pformat
"\nThis info message is printed instead of the arguments when icecream\nfails to find or access source code that's required to parse and analyze.\nThis can happen, for example, when\n\n  - ic() is invoked inside a REPL or interactive shell, e.g. from the\n    command line (CLI) or with python -i.\n\n  - The source code is mangled and/or packaged, e.g. with a project\n    freezer like PyInstaller.\n\n  - The underlying source code changed during execution. See\n    https://stackoverflow.com/a/33175832.\n"
NO_SOURCE_AVAILABLE_WARNING_MESSAGE = 'Failed to access the underlying source code for analysis. Was ic() invoked in a REPL (e.g. from the command line), a frozen application (e.g. packaged with PyInstaller), or did the underlying source code change during execution?'

def callOrValue(obj):
    if False:
        for i in range(10):
            print('nop')
    return obj() if callable(obj) else obj

class Source(executing.Source):

    def get_text_with_indentation(self, node):
        if False:
            return 10
        result = self.asttokens().get_text(node)
        if '\n' in result:
            result = ' ' * node.first_token.start[1] + result
            result = dedent(result)
        result = result.strip()
        return result

def prefixLinesAfterFirst(prefix, s):
    if False:
        while True:
            i = 10
    lines = s.splitlines(True)
    for i in range(1, len(lines)):
        lines[i] = prefix + lines[i]
    return ''.join(lines)

def indented_lines(prefix, string):
    if False:
        return 10
    lines = string.splitlines()
    return [prefix + lines[0]] + [' ' * len(prefix) + line for line in lines[1:]]

def format_pair(prefix, arg, value):
    if False:
        for i in range(10):
            print('nop')
    if arg is _arg_source_missing:
        arg_lines = []
        value_prefix = prefix
    else:
        arg_lines = indented_lines(prefix, arg)
        value_prefix = arg_lines[-1] + ': '
    looksLikeAString = value[0] + value[-1] in ["''", '""']
    if looksLikeAString:
        value = prefixLinesAfterFirst(' ', value)
    value_lines = indented_lines(value_prefix, value)
    lines = arg_lines[:-1] + value_lines
    return '\n'.join(lines)

def singledispatch(func):
    if False:
        return 10
    if 'singledispatch' not in dir(functools):

        def unsupport_py2(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            raise NotImplementedError('functools.singledispatch is missing in ' + sys.version)
        func.register = func.unregister = unsupport_py2
        return func
    func = functools.singledispatch(func)
    closure = dict(zip(func.register.__code__.co_freevars, func.register.__closure__))
    registry = closure['registry'].cell_contents
    dispatch_cache = closure['dispatch_cache'].cell_contents

    def unregister(cls):
        if False:
            print('Hello World!')
        del registry[cls]
        dispatch_cache.clear()
    func.unregister = unregister
    return func

@singledispatch
def argumentToString(obj):
    if False:
        i = 10
        return i + 15
    s = DEFAULT_ARG_TO_STRING_FUNCTION(obj)
    s = s.replace('\\n', '\n')
    return s

class IceCreamDebugger:
    _pairDelimiter = ', '
    lineWrapWidth = DEFAULT_LINE_WRAP_WIDTH
    contextDelimiter = DEFAULT_CONTEXT_DELIMITER

    def __init__(self, prefix=DEFAULT_PREFIX, outputFunction=DEFAULT_OUTPUT_FUNCTION, argToStringFunction=argumentToString, includeContext=False, contextAbsPath=False):
        if False:
            print('Hello World!')
        self.enabled = True
        self.prefix = prefix
        self.includeContext = includeContext
        self.outputFunction = outputFunction
        self.argToStringFunction = argToStringFunction
        self.contextAbsPath = contextAbsPath

    def __call__(self, *args):
        if False:
            return 10
        if self.enabled:
            callFrame = inspect.currentframe().f_back
            self.outputFunction(self._format(callFrame, *args))
        if not args:
            passthrough = None
        elif len(args) == 1:
            passthrough = args[0]
        else:
            passthrough = args
        return passthrough

    def format(self, *args):
        if False:
            while True:
                i = 10
        callFrame = inspect.currentframe().f_back
        out = self._format(callFrame, *args)
        return out

    def _format(self, callFrame, *args):
        if False:
            i = 10
            return i + 15
        prefix = callOrValue(self.prefix)
        context = self._formatContext(callFrame)
        if not args:
            time = self._formatTime()
            out = prefix + context + time
        else:
            if not self.includeContext:
                context = ''
            out = self._formatArgs(callFrame, prefix, context, args)
        return out

    def _formatArgs(self, callFrame, prefix, context, args):
        if False:
            while True:
                i = 10
        callNode = Source.executing(callFrame).node
        if callNode is not None:
            source = Source.for_frame(callFrame)
            sanitizedArgStrs = [source.get_text_with_indentation(arg) for arg in callNode.args]
        else:
            warnings.warn(NO_SOURCE_AVAILABLE_WARNING_MESSAGE, category=RuntimeWarning, stacklevel=4)
            sanitizedArgStrs = [_arg_source_missing] * len(args)
        pairs = list(zip(sanitizedArgStrs, args))
        out = self._constructArgumentOutput(prefix, context, pairs)
        return out

    def _constructArgumentOutput(self, prefix, context, pairs):
        if False:
            i = 10
            return i + 15

        def argPrefix(arg):
            if False:
                return 10
            return '%s: ' % arg
        pairs = [(arg, self.argToStringFunction(val)) for (arg, val) in pairs]
        pairStrs = [val if isLiteral(arg) or arg is _arg_source_missing else argPrefix(arg) + val for (arg, val) in pairs]
        allArgsOnOneLine = self._pairDelimiter.join(pairStrs)
        multilineArgs = len(allArgsOnOneLine.splitlines()) > 1
        contextDelimiter = self.contextDelimiter if context else ''
        allPairs = prefix + context + contextDelimiter + allArgsOnOneLine
        firstLineTooLong = len(allPairs.splitlines()[0]) > self.lineWrapWidth
        if multilineArgs or firstLineTooLong:
            if context:
                lines = [prefix + context] + [format_pair(len(prefix) * ' ', arg, value) for (arg, value) in pairs]
            else:
                arg_lines = [format_pair('', arg, value) for (arg, value) in pairs]
                lines = indented_lines(prefix, '\n'.join(arg_lines))
        else:
            lines = [prefix + context + contextDelimiter + allArgsOnOneLine]
        return '\n'.join(lines)

    def _formatContext(self, callFrame):
        if False:
            print('Hello World!')
        (filename, lineNumber, parentFunction) = self._getContext(callFrame)
        if parentFunction != '<module>':
            parentFunction = '%s()' % parentFunction
        context = '%s:%s in %s' % (filename, lineNumber, parentFunction)
        return context

    def _formatTime(self):
        if False:
            for i in range(10):
                print('nop')
        now = datetime.now()
        formatted = now.strftime('%H:%M:%S.%f')[:-3]
        return ' at %s' % formatted

    def _getContext(self, callFrame):
        if False:
            while True:
                i = 10
        frameInfo = inspect.getframeinfo(callFrame)
        lineNumber = frameInfo.lineno
        parentFunction = frameInfo.function
        filepath = (realpath if self.contextAbsPath else basename)(frameInfo.filename)
        return (filepath, lineNumber, parentFunction)

    def enable(self):
        if False:
            for i in range(10):
                print('nop')
        self.enabled = True

    def disable(self):
        if False:
            while True:
                i = 10
        self.enabled = False

    def configureOutput(self, prefix=_absent, outputFunction=_absent, argToStringFunction=_absent, includeContext=_absent, contextAbsPath=_absent):
        if False:
            for i in range(10):
                print('nop')
        noParameterProvided = all((v is _absent for (k, v) in locals().items() if k != 'self'))
        if noParameterProvided:
            raise TypeError('configureOutput() missing at least one argument')
        if prefix is not _absent:
            self.prefix = prefix
        if outputFunction is not _absent:
            self.outputFunction = outputFunction
        if argToStringFunction is not _absent:
            self.argToStringFunction = argToStringFunction
        if includeContext is not _absent:
            self.includeContext = includeContext
        if contextAbsPath is not _absent:
            self.contextAbsPath = contextAbsPath
ic = IceCreamDebugger()