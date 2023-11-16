"""Compiler tools with improved interactive support.

Provides compilation machinery similar to codeop, but with caching support so
we can provide interactive tracebacks.

Authors
-------
* Robert Kern
* Fernando Perez
* Thomas Kluyver
"""
import __future__
from ast import PyCF_ONLY_AST
import codeop
import functools
import hashlib
import linecache
import operator
import time
from contextlib import contextmanager
PyCF_MASK = functools.reduce(operator.or_, (getattr(__future__, fname).compiler_flag for fname in __future__.all_feature_names))

def code_name(code, number=0):
    if False:
        for i in range(10):
            print('nop')
    ' Compute a (probably) unique name for code for caching.\n\n    This now expects code to be unicode.\n    '
    hash_digest = hashlib.sha1(code.encode('utf-8')).hexdigest()
    return '<ipython-input-{0}-{1}>'.format(number, hash_digest[:12])

class CachingCompiler(codeop.Compile):
    """A compiler that caches code compiled from interactive statements.
    """

    def __init__(self):
        if False:
            return 10
        codeop.Compile.__init__(self)
        self._filename_map = {}

    def ast_parse(self, source, filename='<unknown>', symbol='exec'):
        if False:
            for i in range(10):
                print('nop')
        'Parse code to an AST with the current compiler flags active.\n\n        Arguments are exactly the same as ast.parse (in the standard library),\n        and are passed to the built-in compile function.'
        return compile(source, filename, symbol, self.flags | PyCF_ONLY_AST, 1)

    def reset_compiler_flags(self):
        if False:
            for i in range(10):
                print('nop')
        'Reset compiler flags to default state.'
        self.flags = codeop.PyCF_DONT_IMPLY_DEDENT

    @property
    def compiler_flags(self):
        if False:
            while True:
                i = 10
        'Flags currently active in the compilation process.\n        '
        return self.flags

    def get_code_name(self, raw_code, transformed_code, number):
        if False:
            for i in range(10):
                print('nop')
        "Compute filename given the code, and the cell number.\n\n        Parameters\n        ----------\n        raw_code : str\n            The raw cell code.\n        transformed_code : str\n            The executable Python source code to cache and compile.\n        number : int\n            A number which forms part of the code's name. Used for the execution\n            counter.\n\n        Returns\n        -------\n        The computed filename.\n        "
        return code_name(transformed_code, number)

    def format_code_name(self, name):
        if False:
            for i in range(10):
                print('nop')
        'Return a user-friendly label and name for a code block.\n\n        Parameters\n        ----------\n        name : str\n            The name for the code block returned from get_code_name\n\n        Returns\n        -------\n        A (label, name) pair that can be used in tracebacks, or None if the default formatting should be used.\n        '
        if name in self._filename_map:
            return ('Cell', 'In[%s]' % self._filename_map[name])

    def cache(self, transformed_code, number=0, raw_code=None):
        if False:
            while True:
                i = 10
        "Make a name for a block of code, and cache the code.\n\n        Parameters\n        ----------\n        transformed_code : str\n            The executable Python source code to cache and compile.\n        number : int\n            A number which forms part of the code's name. Used for the execution\n            counter.\n        raw_code : str\n            The raw code before transformation, if None, set to `transformed_code`.\n\n        Returns\n        -------\n        The name of the cached code (as a string). Pass this as the filename\n        argument to compilation, so that tracebacks are correctly hooked up.\n        "
        if raw_code is None:
            raw_code = transformed_code
        name = self.get_code_name(raw_code, transformed_code, number)
        self._filename_map[name] = number
        entry = (len(transformed_code), None, [line + '\n' for line in transformed_code.splitlines()], name)
        linecache.cache[name] = entry
        return name

    @contextmanager
    def extra_flags(self, flags):
        if False:
            return 10
        turn_on_bits = ~self.flags & flags
        self.flags = self.flags | flags
        try:
            yield
        finally:
            self.flags &= ~turn_on_bits

def check_linecache_ipython(*args):
    if False:
        for i in range(10):
            print('nop')
    "Deprecated since IPython 8.6.  Call linecache.checkcache() directly.\n\n    It was already not necessary to call this function directly.  If no\n    CachingCompiler had been created, this function would fail badly.  If\n    an instance had been created, this function would've been monkeypatched\n    into place.\n\n    As of IPython 8.6, the monkeypatching has gone away entirely.  But there\n    were still internal callers of this function, so maybe external callers\n    also existed?\n    "
    import warnings
    warnings.warn('Deprecated Since IPython 8.6, Just call linecache.checkcache() directly.', DeprecationWarning, stacklevel=2)
    linecache.checkcache()