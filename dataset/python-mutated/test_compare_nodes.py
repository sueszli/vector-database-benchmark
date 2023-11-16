from vyper import ast as vy_ast

def test_compare_different_node_clases():
    if False:
        print('Hello World!')
    vyper_ast = vy_ast.parse_to_ast('foo = 42')
    left = vyper_ast.body[0].target
    right = vyper_ast.body[0].value
    assert left != right
    assert not vy_ast.compare_nodes(left, right)

def test_compare_different_nodes_same_class():
    if False:
        i = 10
        return i + 15
    vyper_ast = vy_ast.parse_to_ast('[1, 2]')
    (left, right) = vyper_ast.body[0].value.elements
    assert left != right
    assert not vy_ast.compare_nodes(left, right)

def test_compare_different_nodes_same_value():
    if False:
        while True:
            i = 10
    vyper_ast = vy_ast.parse_to_ast('[1, 1]')
    (left, right) = vyper_ast.body[0].value.elements
    assert left != right
    assert vy_ast.compare_nodes(left, right)

def test_compare_complex_nodes_same_value():
    if False:
        for i in range(10):
            print('nop')
    vyper_ast = vy_ast.parse_to_ast("[{'foo':'bar', 43:[1,2,3]}, {'foo':'bar', 43:[1,2,3]}]")
    (left, right) = vyper_ast.body[0].value.elements
    assert left != right
    assert vy_ast.compare_nodes(left, right)

def test_compare_same_node():
    if False:
        while True:
            i = 10
    vyper_ast = vy_ast.parse_to_ast('42')
    node = vyper_ast.body[0].value
    assert node == node
    assert vy_ast.compare_nodes(node, node)