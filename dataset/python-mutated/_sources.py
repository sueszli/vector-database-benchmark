import ast
import functools
import inspect
from textwrap import dedent
from typing import Any, List, NamedTuple, Optional, Tuple
from torch._C import ErrorReport
from torch._C._jit_tree_views import SourceRangeFactory

def get_source_lines_and_file(obj: Any, error_msg: Optional[str]=None) -> Tuple[List[str], int, Optional[str]]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Wrapper around inspect.getsourcelines and inspect.getsourcefile.\n\n    Returns: (sourcelines, file_lino, filename)\n    '
    filename = None
    try:
        filename = inspect.getsourcefile(obj)
        (sourcelines, file_lineno) = inspect.getsourcelines(obj)
    except OSError as e:
        msg = f"Can't get source for {obj}. TorchScript requires source access in order to carry out compilation, make sure original .py files are available."
        if error_msg:
            msg += '\n' + error_msg
        raise OSError(msg) from e
    return (sourcelines, file_lineno, filename)

def normalize_source_lines(sourcelines: List[str]) -> List[str]:
    if False:
        return 10
    "\n    This helper function accepts a list of source lines. It finds the\n    indentation level of the function definition (`def`), then it indents\n    all lines in the function body to a point at or greater than that\n    level. This allows for comments and continued string literals that\n    are at a lower indentation than the rest of the code.\n    Args:\n        sourcelines: function source code, separated into lines by\n                        the '\n' character\n    Returns:\n        A list of source lines that have been correctly aligned\n    "

    def remove_prefix(text, prefix):
        if False:
            print('Hello World!')
        return text[text.startswith(prefix) and len(prefix):]
    idx = None
    for (i, l) in enumerate(sourcelines):
        if l.lstrip().startswith('def'):
            idx = i
            break
    if idx is None:
        return sourcelines
    fn_def = sourcelines[idx]
    whitespace = fn_def.split('def')[0]
    aligned_prefix = [whitespace + remove_prefix(s, whitespace) for s in sourcelines[:idx]]
    aligned_suffix = [whitespace + remove_prefix(s, whitespace) for s in sourcelines[idx + 1:]]
    aligned_prefix.append(fn_def)
    return aligned_prefix + aligned_suffix

class SourceContext(SourceRangeFactory):

    def __init__(self, source, filename, file_lineno, leading_whitespace_len, uses_true_division=True, funcname=None):
        if False:
            print('Hello World!')
        super().__init__(source, filename, file_lineno, leading_whitespace_len)
        self.uses_true_division = uses_true_division
        self.filename = filename
        self.funcname = funcname

@functools.lru_cache(maxsize=None)
def make_source_context(*args):
    if False:
        return 10
    return SourceContext(*args)

def fake_range():
    if False:
        return 10
    return SourceContext('', None, 0, 0).make_raw_range(0, 1)

class ParsedDef(NamedTuple):
    ast: ast.Module
    ctx: SourceContext
    source: str
    filename: Optional[str]
    file_lineno: int

def parse_def(fn):
    if False:
        while True:
            i = 10
    (sourcelines, file_lineno, filename) = get_source_lines_and_file(fn, ErrorReport.call_stack())
    sourcelines = normalize_source_lines(sourcelines)
    source = ''.join(sourcelines)
    dedent_src = dedent(source)
    py_ast = ast.parse(dedent_src)
    if len(py_ast.body) != 1 or not isinstance(py_ast.body[0], ast.FunctionDef):
        raise RuntimeError(f'Expected a single top-level function: {filename}:{file_lineno}')
    leading_whitespace_len = len(source.split('\n', 1)[0]) - len(dedent_src.split('\n', 1)[0])
    ctx = make_source_context(source, filename, file_lineno, leading_whitespace_len, True, fn.__name__)
    return ParsedDef(py_ast, ctx, source, filename, file_lineno)