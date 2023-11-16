from __future__ import annotations
import ast
from pre_commit_hooks.debug_statement_hook import Debug
from pre_commit_hooks.debug_statement_hook import DebugStatementParser
from pre_commit_hooks.debug_statement_hook import main
from testing.util import get_resource_path

def test_no_breakpoints():
    if False:
        i = 10
        return i + 15
    visitor = DebugStatementParser()
    visitor.visit(ast.parse('import os\nfrom foo import bar\n'))
    assert visitor.breakpoints == []

def test_finds_debug_import_attribute_access():
    if False:
        print('Hello World!')
    visitor = DebugStatementParser()
    visitor.visit(ast.parse('import ipdb; ipdb.set_trace()'))
    assert visitor.breakpoints == [Debug(1, 0, 'ipdb', 'imported')]

def test_finds_debug_import_from_import():
    if False:
        return 10
    visitor = DebugStatementParser()
    visitor.visit(ast.parse('from pudb import set_trace; set_trace()'))
    assert visitor.breakpoints == [Debug(1, 0, 'pudb', 'imported')]

def test_finds_breakpoint():
    if False:
        return 10
    visitor = DebugStatementParser()
    visitor.visit(ast.parse('breakpoint()'))
    assert visitor.breakpoints == [Debug(1, 0, 'breakpoint', 'called')]

def test_returns_one_for_failing_file(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    f_py = tmpdir.join('f.py')
    f_py.write('def f():\n    import pdb; pdb.set_trace()')
    ret = main([str(f_py)])
    assert ret == 1

def test_returns_zero_for_passing_file():
    if False:
        return 10
    ret = main([__file__])
    assert ret == 0

def test_syntaxerror_file():
    if False:
        print('Hello World!')
    ret = main([get_resource_path('cannot_parse_ast.notpy')])
    assert ret == 1

def test_non_utf8_file(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    f_py = tmpdir.join('f.py')
    f_py.write_binary('# -*- coding: cp1252 -*-\nx = "€"\n'.encode('cp1252'))
    assert main((str(f_py),)) == 0

def test_py37_breakpoint(tmpdir, capsys):
    if False:
        for i in range(10):
            print('nop')
    f_py = tmpdir.join('f.py')
    f_py.write('def f():\n    breakpoint()\n')
    assert main((str(f_py),)) == 1
    (out, _) = capsys.readouterr()
    assert out == f'{f_py}:2:4: breakpoint called\n'