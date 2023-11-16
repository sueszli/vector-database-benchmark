"""Utilities to compile possibly incomplete Python source code.

This module provides two interfaces, broadly similar to the builtin
function compile(), which take program text, a filename and a 'mode'
and:

- Return code object if the command is complete and valid
- Return None if the command is incomplete
- Raise SyntaxError, ValueError or OverflowError if the command is a
  syntax error (OverflowError and ValueError can be produced by
  malformed literals).

The two interfaces are:

compile_command(source, filename, symbol):

    Compiles a single command in the manner described above.

CommandCompiler():

    Instances of this class have __call__ methods identical in
    signature to compile_command; the difference is that if the
    instance compiles program text containing a __future__ statement,
    the instance 'remembers' and compiles all subsequent program texts
    with the statement in force.

The module also provides another class:

Compile():

    Instances of this class act like the built-in function compile,
    but with 'memory' in the sense described above.
"""
import __future__
import warnings
_features = [getattr(__future__, fname) for fname in __future__.all_feature_names]
__all__ = ['compile_command', 'Compile', 'CommandCompiler']
PyCF_DONT_IMPLY_DEDENT = 512
PyCF_ALLOW_INCOMPLETE_INPUT = 16384

def _maybe_compile(compiler, source, filename, symbol):
    if False:
        i = 10
        return i + 15
    for line in source.split('\n'):
        line = line.strip()
        if line and line[0] != '#':
            break
    else:
        if symbol != 'eval':
            source = 'pass'
    try:
        return compiler(source, filename, symbol)
    except SyntaxError:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        try:
            compiler(source + '\n', filename, symbol)
        except SyntaxError as e:
            if 'incomplete input' in str(e):
                return None
            raise

def _is_syntax_error(err1, err2):
    if False:
        i = 10
        return i + 15
    rep1 = repr(err1)
    rep2 = repr(err2)
    if 'was never closed' in rep1 and 'was never closed' in rep2:
        return False
    if rep1 == rep2:
        return True
    return False

def _compile(source, filename, symbol):
    if False:
        i = 10
        return i + 15
    return compile(source, filename, symbol, PyCF_DONT_IMPLY_DEDENT | PyCF_ALLOW_INCOMPLETE_INPUT)

def compile_command(source, filename='<input>', symbol='single'):
    if False:
        while True:
            i = 10
    'Compile a command and determine whether it is incomplete.\n\n    Arguments:\n\n    source -- the source string; may contain \\n characters\n    filename -- optional filename from which source was read; default\n                "<input>"\n    symbol -- optional grammar start symbol; "single" (default), "exec"\n              or "eval"\n\n    Return value / exceptions raised:\n\n    - Return a code object if the command is complete and valid\n    - Return None if the command is incomplete\n    - Raise SyntaxError, ValueError or OverflowError if the command is a\n      syntax error (OverflowError and ValueError can be produced by\n      malformed literals).\n    '
    return _maybe_compile(_compile, source, filename, symbol)

class Compile:
    """Instances of this class behave much like the built-in compile
    function, but if one is used to compile text containing a future
    statement, it "remembers" and compiles all subsequent program texts
    with the statement in force."""

    def __init__(self):
        if False:
            return 10
        self.flags = PyCF_DONT_IMPLY_DEDENT | PyCF_ALLOW_INCOMPLETE_INPUT

    def __call__(self, source, filename, symbol):
        if False:
            while True:
                i = 10
        codeob = compile(source, filename, symbol, self.flags, True)
        for feature in _features:
            if codeob.co_flags & feature.compiler_flag:
                self.flags |= feature.compiler_flag
        return codeob

class CommandCompiler:
    """Instances of this class have __call__ methods identical in
    signature to compile_command; the difference is that if the
    instance compiles program text containing a __future__ statement,
    the instance 'remembers' and compiles all subsequent program texts
    with the statement in force."""

    def __init__(self):
        if False:
            return 10
        self.compiler = Compile()

    def __call__(self, source, filename='<input>', symbol='single'):
        if False:
            while True:
                i = 10
        'Compile a command and determine whether it is incomplete.\n\n        Arguments:\n\n        source -- the source string; may contain \\n characters\n        filename -- optional filename from which source was read;\n                    default "<input>"\n        symbol -- optional grammar start symbol; "single" (default) or\n                  "eval"\n\n        Return value / exceptions raised:\n\n        - Return a code object if the command is complete and valid\n        - Return None if the command is incomplete\n        - Raise SyntaxError, ValueError or OverflowError if the command is a\n          syntax error (OverflowError and ValueError can be produced by\n          malformed literals).\n        '
        return _maybe_compile(self.compiler, source, filename, symbol)