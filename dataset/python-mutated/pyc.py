"""Functions for generating, reading and parsing pyc."""
import copy
from pycnite import pyc
from pytype import utils
from pytype.pyc import compiler
CompileError = compiler.CompileError

def parse_pyc_string(data):
    if False:
        while True:
            i = 10
    'Parse pyc data from a string.\n\n  Args:\n    data: pyc data\n\n  Returns:\n    An instance of pycnite.types.CodeTypeBase.\n  '
    return pyc.loads(data)

class AdjustFilename:
    """Visitor for changing co_filename in a code object."""

    def __init__(self, filename):
        if False:
            i = 10
            return i + 15
        self.filename = filename

    def visit_code(self, code):
        if False:
            return 10
        code.co_filename = self.filename
        return code

def compile_src(src, filename, python_version, python_exe, mode='exec'):
    if False:
        while True:
            i = 10
    'Compile a string to pyc, and then load and parse the pyc.\n\n  Args:\n    src: Python source code.\n    filename: The filename the sourcecode is from.\n    python_version: Python version, (major, minor).\n    python_exe: The path to Python interpreter.\n    mode: "exec", "eval" or "single".\n\n  Returns:\n    An instance of pycnite.types.CodeTypeBase.\n\n  Raises:\n    UsageError: If python_exe and python_version are mismatched.\n  '
    pyc_data = compiler.compile_src_string_to_pyc_string(src, filename, python_version, python_exe, mode)
    code = parse_pyc_string(pyc_data)
    if code.python_version != python_version:
        raise utils.UsageError('python_exe version %s does not match python version %s' % (utils.format_version(code.python_version), utils.format_version(python_version)))
    visit(code, AdjustFilename(filename))
    return code
_VISIT_CACHE = {}

def visit(c, visitor):
    if False:
        i = 10
        return i + 15
    'Recursively process constants in a pyc using a visitor.'
    if hasattr(c, 'co_consts'):
        k = (id(c), visitor)
        if k not in _VISIT_CACHE:
            new_consts = []
            changed = False
            for const in c.co_consts:
                new_const = visit(const, visitor)
                changed |= new_const is not const
                new_consts.append(new_const)
            if changed:
                c = copy.copy(c)
                c.co_consts = new_consts
            _VISIT_CACHE[k] = visitor.visit_code(c)
        return _VISIT_CACHE[k]
    else:
        return c