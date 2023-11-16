"""Run doctests for tensorflow."""
import ast
import doctest
import os
import re
import textwrap
from typing import Any, Callable, Dict, Iterable, Optional
import astor
from tensorflow.tools.docs import tf_doctest_lib

def load_from_files(files, globs: Optional[Dict[str, Any]]=None, set_up: Optional[Callable[[Any], None]]=None, tear_down: Optional[Callable[[Any], None]]=None) -> doctest.DocFileSuite:
    if False:
        for i in range(10):
            print('nop')
    'Creates a doctest suite from the files list.\n\n  Args:\n    files: A list of file paths to test.\n    globs: The global namespace the tests are run in.\n    set_up: Run before each test, receives the test as argument.\n    tear_down: Run after each test, receives the test as argument.\n\n  Returns:\n    A DocFileSuite containing the tests.\n  '
    if globs is None:
        globs = {}
    files = [os.fspath(f) for f in files]
    globs['_print_if_not_none'] = _print_if_not_none
    return doctest.DocFileSuite(*files, module_relative=False, parser=FencedCellParser(fence_label='python'), globs=globs, setUp=set_up, tearDown=tear_down, checker=FencedCellOutputChecker(), optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE | doctest.IGNORE_EXCEPTION_DETAIL | doctest.DONT_ACCEPT_BLANKLINE)

class FencedCellOutputChecker(tf_doctest_lib.TfDoctestOutputChecker):
    """TfDoctestChecker with a different warning message."""
    MESSAGE = textwrap.dedent('\n\n        ##############################################################\n        # Check the documentation (go/g3doctest) on how to write\n        # testable g3docs.\n        ##############################################################\n        ')

class FencedCellParser(doctest.DocTestParser):
    """Implements test parsing for ``` fenced cells.

  https://docs.python.org/3/library/doctest.html#doctestparser-objects

  The `get_examples` method receives a string and returns an
  iterable of `doctest.Example` objects.
  """
    patched = False

    def __init__(self, fence_label='python'):
        if False:
            while True:
                i = 10
        super().__init__()
        if not self.patched:
            doctest.compile = _patch_compile
            print(textwrap.dedent("\n          *********************************************************************\n          * Caution: `fenced_doctest` patches `doctest.compile` don't use this\n          *   in the same binary as any other doctests.\n          *********************************************************************\n          "))
            type(self).patched = True
        no_fence = '(.(?<!```))*?'
        self.fence_cell_re = re.compile(f'\n        ^(                             # After a newline\n            \\s*```\\s*({fence_label})\\n   # Open a labeled ``` fence\n            (?P<doctest>{no_fence})      # Match anything except a closing fence\n            \\n\\s*```\\s*(\\n|$)            # Close the fence.\n        )\n        (                              # Optional!\n            [\\s\\n]*                      # Any number of blank lines.\n            ```\\s*\\n                     # Open ```\n            (?P<output>{no_fence})       # Anything except a closing fence\n            \\n\\s*```                     # Close the fence.\n        )?\n        ', re.MULTILINE | re.DOTALL | re.VERBOSE)

    def get_examples(self, string: str, name: str='<string>') -> Iterable[doctest.Example]:
        if False:
            print('Hello World!')
        if re.search('<!--.*?doctest.*?skip.*?all.*?-->', string, re.IGNORECASE):
            return
        for match in self.fence_cell_re.finditer(string):
            if re.search('doctest.*skip', match.group(0), re.IGNORECASE):
                continue
            groups = match.groupdict()
            source = textwrap.dedent(groups['doctest'])
            want = groups['output']
            if want is not None:
                want = textwrap.dedent(want)
            yield doctest.Example(lineno=string[:match.start()].count('\n') + 1, source=source, want=want)

def _print_if_not_none(obj):
    if False:
        print('Hello World!')
    'Print like a notebook: Show the repr if the object is not None.\n\n  `_patch_compile` Uses this on the final expression in each cell.\n\n  This way the outputs feel like notebooks.\n\n  Args:\n    obj: the object to print.\n  '
    if obj is not None:
        print(repr(obj))

def _patch_compile(source, filename, mode, flags=0, dont_inherit=False, optimize=-1):
    if False:
        i = 10
        return i + 15
    'Patch `doctest.compile` to make doctest to behave like a notebook.\n\n  Default settings for doctest are configured to run like a repl: one statement\n  at a time. The doctest source uses `compile(..., mode="single")`\n\n  So to let doctest act like a notebook:\n\n  1. We need `mode="exec"` (easy)\n  2. We need the last expression to be printed (harder).\n\n  To print the last expression, just wrap the last expression in\n  `_print_if_not_none(expr)`. To detect the last expression use `AST`.\n  If the last node is an expression modify the ast to call\n  `_print_if_not_none` on it, convert the ast back to source and compile that.\n\n  https://docs.python.org/3/library/functions.html#compile\n\n  Args:\n    source: Can either be a normal string, a byte string, or an AST object.\n    filename: Argument should give the file from which the code was read; pass\n      some recognizable value if it wasn’t read from a file (\'<string>\' is\n      commonly used).\n    mode: [Ignored] always use exec.\n    flags: Compiler options.\n    dont_inherit: Compiler options.\n    optimize: Compiler options.\n\n  Returns:\n    The resulting code object.\n  '
    del filename
    del mode
    source_ast = ast.parse(source)
    final = source_ast.body[-1]
    if isinstance(final, ast.Expr):
        print_it = ast.Expr(lineno=-1, col_offset=-1, value=ast.Call(func=ast.Name(id='_print_if_not_none', ctx=ast.Load(), lineno=-1, col_offset=-1), lineno=-1, col_offset=-1, args=[final], keywords=[]))
        source_ast.body[-1] = print_it
        source = astor.to_source(source_ast)
    return compile(source, filename='dummy.py', mode='exec', flags=flags, dont_inherit=dont_inherit, optimize=optimize)