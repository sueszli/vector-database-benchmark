"""Tests the xonsh lexer."""
import os
import pytest
from xonsh.wizard import FileInserter, Message, Pass, PrettyFormatter, Question, StateVisitor, Wizard
TREE0 = Wizard(children=[Pass(), Message(message='yo')])
TREE1 = Question('wakka?', {'jawaka': Pass()})

def test_pretty_format_tree0():
    if False:
        while True:
            i = 10
    exp = "Wizard(children=[\n Pass(),\n Message('yo')\n])"
    obs = PrettyFormatter(TREE0).visit()
    assert exp == obs
    assert exp == str(TREE0)
    assert exp.replace('\n', '') == repr(TREE0)

def test_pretty_format_tree1():
    if False:
        while True:
            i = 10
    exp = "Question(\n question='wakka?',\n responses={\n  'jawaka': Pass()\n }\n)"
    obs = PrettyFormatter(TREE1).visit()
    assert exp == obs
    assert exp == str(TREE1)
    assert exp.replace('\n', '') == repr(TREE1)

def test_state_visitor_store():
    if False:
        for i in range(10):
            print('nop')
    exp = {'rick': [{}, {}, {'and': 'morty'}]}
    sv = StateVisitor()
    sv.store('/rick/2/and', 'morty')
    obs = sv.state
    assert exp == obs
    exp['rick'][1]['mr'] = 'meeseeks'
    sv.store('/rick/-2/mr', 'meeseeks')
    assert exp == obs
    flat_exp = {'/': {'rick': [{}, {'mr': 'meeseeks'}, {'and': 'morty'}]}, '/rick/': [{}, {'mr': 'meeseeks'}, {'and': 'morty'}], '/rick/0/': {}, '/rick/1/': {'mr': 'meeseeks'}, '/rick/1/mr': 'meeseeks', '/rick/2/': {'and': 'morty'}, '/rick/2/and': 'morty'}
    flat_obs = sv.flatten()
    assert flat_exp == flat_obs

def dump_xonfig_env_mock(path, value):
    if False:
        for i in range(10):
            print('nop')
    name = os.path.basename(path.rstrip('/'))
    return f'${name} = {value!r}'

def test_tuple_store_and_write():
    if False:
        for i in range(10):
            print('nop')
    sv = StateVisitor()
    sv.store('/env/XONSH_HISTORY_SIZE', (1073741824, 'b'))
    dump_rules = {'/': None, '/env/': None, '/env/*': dump_xonfig_env_mock, '/env/*/[0-9]*': None}
    fi = FileInserter(prefix='# XONSH WIZARD START', suffix='# XONSH WIZARD END', dump_rules=dump_rules, default_file=None, check=False, ask_filename=False)
    exp = "# XONSH WIZARD START\n$XONSH_HISTORY_SIZE = (1073741824, 'b')\n# XONSH WIZARD END\n"
    obs = fi.dumps(sv.flatten())
    assert exp == obs