from __future__ import annotations
import ast
import pytest
from pre_commit_hooks.check_builtin_literals import Call
from pre_commit_hooks.check_builtin_literals import main
from pre_commit_hooks.check_builtin_literals import Visitor
BUILTIN_CONSTRUCTORS = 'import builtins\n\nc1 = complex()\nd1 = dict()\nf1 = float()\ni1 = int()\nl1 = list()\ns1 = str()\nt1 = tuple()\n\nc2 = builtins.complex()\nd2 = builtins.dict()\nf2 = builtins.float()\ni2 = builtins.int()\nl2 = builtins.list()\ns2 = builtins.str()\nt2 = builtins.tuple()\n'
BUILTIN_LITERALS = "c1 = 0j\nd1 = {}\nf1 = 0.0\ni1 = 0\nl1 = []\ns1 = ''\nt1 = ()\n"

@pytest.fixture
def visitor():
    if False:
        return 10
    return Visitor()

@pytest.mark.parametrize(('expression', 'calls'), [('x[0]()', []), ('0j', []), ('complex()', [Call('complex', 1, 0)]), ('complex(0, 0)', []), ("complex('0+0j')", []), ('builtins.complex()', []), ('0.0', []), ('float()', [Call('float', 1, 0)]), ("float('0.0')", []), ('builtins.float()', []), ('0', []), ('int()', [Call('int', 1, 0)]), ("int('0')", []), ('builtins.int()', []), ('[]', []), ('list()', [Call('list', 1, 0)]), ("list('abc')", []), ("list([c for c in 'abc'])", []), ("list(c for c in 'abc')", []), ('builtins.list()', []), ("''", []), ('str()', [Call('str', 1, 0)]), ("str('0')", []), ('builtins.str()', []), ('()', []), ('tuple()', [Call('tuple', 1, 0)]), ("tuple('abc')", []), ("tuple([c for c in 'abc'])", []), ("tuple(c for c in 'abc')", []), ('builtins.tuple()', [])])
def test_non_dict_exprs(visitor, expression, calls):
    if False:
        print('Hello World!')
    visitor.visit(ast.parse(expression))
    assert visitor.builtin_type_calls == calls

@pytest.mark.parametrize(('expression', 'calls'), [('{}', []), ('dict()', [Call('dict', 1, 0)]), ('dict(a=1, b=2, c=3)', []), ("dict(**{'a': 1, 'b': 2, 'c': 3})", []), ("dict([(k, v) for k, v in [('a', 1), ('b', 2), ('c', 3)]])", []), ("dict((k, v) for k, v in [('a', 1), ('b', 2), ('c', 3)])", []), ('builtins.dict()', [])])
def test_dict_allow_kwargs_exprs(visitor, expression, calls):
    if False:
        i = 10
        return i + 15
    visitor.visit(ast.parse(expression))
    assert visitor.builtin_type_calls == calls

@pytest.mark.parametrize(('expression', 'calls'), [('dict()', [Call('dict', 1, 0)]), ('dict(a=1, b=2, c=3)', [Call('dict', 1, 0)]), ("dict(**{'a': 1, 'b': 2, 'c': 3})", [Call('dict', 1, 0)]), ('builtins.dict()', [])])
def test_dict_no_allow_kwargs_exprs(expression, calls):
    if False:
        i = 10
        return i + 15
    visitor = Visitor(allow_dict_kwargs=False)
    visitor.visit(ast.parse(expression))
    assert visitor.builtin_type_calls == calls

def test_ignore_constructors():
    if False:
        while True:
            i = 10
    visitor = Visitor(ignore=('complex', 'dict', 'float', 'int', 'list', 'str', 'tuple'))
    visitor.visit(ast.parse(BUILTIN_CONSTRUCTORS))
    assert visitor.builtin_type_calls == []

def test_failing_file(tmpdir):
    if False:
        while True:
            i = 10
    f = tmpdir.join('f.py')
    f.write(BUILTIN_CONSTRUCTORS)
    rc = main([str(f)])
    assert rc == 1

def test_passing_file(tmpdir):
    if False:
        while True:
            i = 10
    f = tmpdir.join('f.py')
    f.write(BUILTIN_LITERALS)
    rc = main([str(f)])
    assert rc == 0

def test_failing_file_ignore_all(tmpdir):
    if False:
        i = 10
        return i + 15
    f = tmpdir.join('f.py')
    f.write(BUILTIN_CONSTRUCTORS)
    rc = main(['--ignore=complex,dict,float,int,list,str,tuple', str(f)])
    assert rc == 0