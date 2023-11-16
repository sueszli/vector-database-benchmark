import pytest

def test_ok():
    if False:
        print('Hello World!')
    assert something
    assert something or something_else
    assert something or (something_else and something_third)
    assert not (something and something_else)
    assert something, 'something message'
    assert something or (something_else and something_third), 'another message'

def test_error():
    if False:
        return 10
    assert something and something_else
    assert something and something_else and something_third
    assert something and (not something_else)
    assert something and (something_else or something_third)
    assert not something and something_else
    assert not (something or something_else)
    assert not (something or something_else or something_third)
    assert something and something_else == 'error\n    message\n    '
    assert something and something_else == 'error\nmessage\n'
    assert not (a or not (b or c))
    assert not (a or not (b and c))
    assert something and something_else, 'error message'
    assert not (something or (something_else and something_third)), 'with message'
    assert not (something or (something_else and something_third))
assert something
assert something and something_else
assert something and something_else and something_third

def test_multiline():
    if False:
        return 10
    assert something and something_else
    x = 1
    x = 1
    assert something and something_else
    x = 1
    assert something and something_else

def test_parenthesized_not():
    if False:
        print('Hello World!')
    assert not (self.find_graph_output(node.output[0]) or self.find_graph_input(node.input[0]) or self.find_graph_output(node.input[0]))
    assert not (self.find_graph_output(node.output[0]) or self.find_graph_input(node.input[0]) or self.find_graph_output(node.input[0]))
    assert not self.find_graph_output(node.output[0]) or self.find_graph_input(node.input[0])