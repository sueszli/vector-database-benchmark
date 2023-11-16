from hscommon.testutil import eq_
from hscommon.gui.tree import Tree, Node

def tree_with_some_nodes():
    if False:
        return 10
    t = Tree()
    t.append(Node('foo'))
    t.append(Node('bar'))
    t.append(Node('baz'))
    t[0].append(Node('sub1'))
    t[0].append(Node('sub2'))
    return t

def test_selection():
    if False:
        while True:
            i = 10
    t = tree_with_some_nodes()
    assert t.selected_node is None
    eq_(t.selected_nodes, [])
    assert t.selected_path is None
    eq_(t.selected_paths, [])

def test_select_one_node():
    if False:
        print('Hello World!')
    t = tree_with_some_nodes()
    t.selected_node = t[0][0]
    assert t.selected_node is t[0][0]
    eq_(t.selected_nodes, [t[0][0]])
    eq_(t.selected_path, [0, 0])
    eq_(t.selected_paths, [[0, 0]])

def test_select_one_path():
    if False:
        return 10
    t = tree_with_some_nodes()
    t.selected_path = [0, 1]
    assert t.selected_node is t[0][1]

def test_select_multiple_nodes():
    if False:
        while True:
            i = 10
    t = tree_with_some_nodes()
    t.selected_nodes = [t[0], t[1]]
    eq_(t.selected_paths, [[0], [1]])

def test_select_multiple_paths():
    if False:
        return 10
    t = tree_with_some_nodes()
    t.selected_paths = [[0], [1]]
    eq_(t.selected_nodes, [t[0], t[1]])

def test_select_none_path():
    if False:
        return 10
    t = Tree()
    t.selected_path = None
    assert t.selected_path is None

def test_select_none_node():
    if False:
        print('Hello World!')
    t = Tree()
    t.selected_node = None
    eq_(t.selected_nodes, [])

def test_clear_removes_selection():
    if False:
        return 10
    t = tree_with_some_nodes()
    t.selected_path = [0]
    t.clear()
    assert t.selected_node is None

def test_selection_override():
    if False:
        print('Hello World!')

    class MyTree(Tree):
        called = False

        def _select_nodes(self, nodes):
            if False:
                print('Hello World!')
            self.called = True
    t = MyTree()
    t.selected_paths = []
    assert t.called
    t.called = False
    t.selected_node = None
    assert t.called

def test_findall():
    if False:
        while True:
            i = 10
    t = tree_with_some_nodes()
    r = t.findall(lambda n: n.name.startswith('sub'))
    eq_(set(r), {t[0][0], t[0][1]})

def test_findall_dont_include_self():
    if False:
        print('Hello World!')
    t = tree_with_some_nodes()
    del t._name
    r = t.findall(lambda n: not n.name.startswith('sub'), include_self=False)
    eq_(set(r), {t[0], t[1], t[2]})

def test_find_dont_include_self():
    if False:
        return 10
    t = tree_with_some_nodes()
    del t._name
    r = t.find(lambda n: not n.name.startswith('sub'), include_self=False)
    assert r is t[0]

def test_find_none():
    if False:
        while True:
            i = 10
    t = Tree()
    assert t.find(lambda n: False) is None