from pathlib import Path
from vyper import ast as vy_ast

def test_all_nodes_have_lineno():
    if False:
        for i in range(10):
            print('nop')
    for path in Path('.').glob('examples/**/*.vy'):
        with path.open() as fp:
            source = fp.read()
            vyper_ast = vy_ast.parse_to_ast(source)
            for item in vyper_ast.get_descendants():
                assert hasattr(item, 'lineno')
                assert item.lineno > 0

def test_returns_all_descendants():
    if False:
        while True:
            i = 10
    vyper_ast = vy_ast.parse_to_ast('[1, 2, (3, 4, 5, 6), 7]')
    descendants = vyper_ast.get_descendants()
    assert vyper_ast.body[0] in descendants
    for node in vyper_ast.body[0].value.elements:
        assert node in descendants
    for node in vyper_ast.body[0].value.elements[2].elements:
        assert node in descendants

def test_type_filter():
    if False:
        i = 10
        return i + 15
    vyper_ast = vy_ast.parse_to_ast("[1, (2, (3, (4, 5.0), 'six')), 7, 0x08]")
    descendants = vyper_ast.get_descendants(vy_ast.Int)
    assert len(descendants) == 5
    assert not next((i for i in descendants if not isinstance(i, vy_ast.Int)), False)

def test_dict_filter():
    if False:
        while True:
            i = 10
    node = vy_ast.parse_to_ast('[foo, (foo(), bar), bar()]').body[0].value
    assert node.get_descendants(filters={'func.id': 'foo'}) == [node.elements[1].elements[0]]

def test_include_self():
    if False:
        return 10
    vyper_ast = vy_ast.parse_to_ast('1 + 2')
    node = vyper_ast.body[0].value
    descendants = node.get_descendants(vy_ast.BinOp, include_self=True)
    assert descendants == [node]

def test_include_self_wrong_type():
    if False:
        while True:
            i = 10
    vyper_ast = vy_ast.parse_to_ast('1 + 2')
    descendants = vyper_ast.get_descendants(vy_ast.Int, include_self=True)
    assert vyper_ast not in descendants

def test_order():
    if False:
        i = 10
        return i + 15
    node = vy_ast.parse_to_ast('[(1 + (2 - 3)) / 4 ** 5, 6 - (7 / -(8 % 9)), 0]')
    node = node.body[0].value
    values = [i.value for i in node.get_descendants(vy_ast.Int)]
    assert values == [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]

def test_order_reversed():
    if False:
        i = 10
        return i + 15
    node = vy_ast.parse_to_ast('[(1 + (2 - 3)) / 4 ** 5, 6 - (7 / -(8 % 9)), 0]')
    node = node.body[0].value
    values = [i.value for i in node.get_descendants(vy_ast.Int, reverse=True)]
    assert values == [0, 9, 8, 7, 6, 5, 4, 3, 2, 1]