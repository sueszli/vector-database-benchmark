import ast
import dis
from compiler.dis_stable import Disassembler
from compiler.pycodegen import compile as py_compile
from io import StringIO
from os import path
from tokenize import detect_encoding
from unittest import TestCase
from .common import glob_test
N_SBS_TEST_CLASSES = 10
IGNORE_PATTERNS = ('lib2to3/tests/data', 'test/badsyntax_', 'test/bad_coding', 'test/test_compiler/testcorpus')
SbsCompileTests = []
for i in range(N_SBS_TEST_CLASSES):
    class_name = f'SbsCompileTests{i}'
    new_class = type(class_name, (TestCase,), {})
    SbsCompileTests.append(new_class)
    globals()[class_name] = new_class

def add_test(modname, fname):
    if False:
        print('Hello World!')
    assert fname.startswith(libpath + '/')
    for p in IGNORE_PATTERNS:
        if p in fname:
            return
    modname = path.relpath(fname, REPO_ROOT)

    def test_stdlib(self):
        if False:
            i = 10
            return i + 15
        with open(fname, 'rb') as inp:
            (encoding, _lines) = detect_encoding(inp.readline)
            code = b''.join(_lines + inp.readlines()).decode(encoding)
            node = ast.parse(code, modname, 'exec')
            node.filename = modname
            orig = compile(node, modname, 'exec')
            origdump = StringIO()
            Disassembler().dump_code(orig, origdump)
            codeobj = py_compile(node, modname, 'exec')
            newdump = StringIO()
            Disassembler().dump_code(codeobj, newdump)
            try:
                self.assertEqual(origdump.getvalue().split('\n'), newdump.getvalue().split('\n'))
            except AssertionError:
                with open('c_compiler_output.txt', 'w') as f:
                    f.write(origdump.getvalue())
                with open('py_compiler_output.txt', 'w') as f:
                    f.write(newdump.getvalue())
                raise
    name = 'test_stdlib_' + modname.replace('/', '_')[:-3]
    test_stdlib.__name__ = name
    n = hash(name) % N_SBS_TEST_CLASSES
    setattr(SbsCompileTests[n], test_stdlib.__name__, test_stdlib)
REPO_ROOT = path.join(path.dirname(__file__), '..', '..', '..')
libpath = path.join(REPO_ROOT, 'Lib')
if path.exists(libpath):
    glob_test(libpath, '**/*.py', add_test)
else:
    libpath = LIB_PATH = path.dirname(dis.__file__)
    glob_test(LIB_PATH, '**/*.py', add_test)
    IGNORE_PATTERNS = tuple((pattern.replace('test/test_compiler/', 'test_compiler/') for pattern in IGNORE_PATTERNS))