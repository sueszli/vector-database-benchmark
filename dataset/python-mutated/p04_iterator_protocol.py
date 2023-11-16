"""
Topic: 自定义迭代器协议
Desc : 
"""

class Node:

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        self._value = value
        self._children = []

    def __repr__(self):
        if False:
            return 10
        return 'Node({!r})'.format(self._value)

    def add_child(self, node):
        if False:
            print('Hello World!')
        self._children.append(node)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self._children)

    def depth_first(self):
        if False:
            i = 10
            return i + 15
        yield self
        for c in self:
            yield from c.depth_first()

class Node2:

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        self._value = value
        self._children = []

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'Node({!r})'.format(self._value)

    def add_child(self, node):
        if False:
            print('Hello World!')
        self._children.append(node)

    def __iter__(self):
        if False:
            i = 10
            return i + 15
        return iter(self._children)

    def depth_first(self):
        if False:
            print('Hello World!')
        return DepthFirstIterator(self)

class DepthFirstIterator(object):
    """
    Depth-first traversal
    """

    def __init__(self, start_node):
        if False:
            for i in range(10):
                print('nop')
        self._node = start_node
        self._children_iter = None
        self._child_iter = None

    def __iter__(self):
        if False:
            return 10
        return self

    def __next__(self):
        if False:
            print('Hello World!')
        if self._children_iter is None:
            self._children_iter = iter(self._node)
            return self._node
        elif self._child_iter:
            try:
                nextchild = next(self._child_iter)
                return nextchild
            except StopIteration:
                self._child_iter = None
                return next(self)
        else:
            self._child_iter = next(self._children_iter).depth_first()
            return next(self)
if __name__ == '__main__':
    root = Node(0)
    child1 = Node(1)
    child2 = Node(2)
    root.add_child(child1)
    root.add_child(child2)
    child1.add_child(Node(3))
    child1.add_child(Node(4))
    child2.add_child(Node(5))
    for ch in root.depth_first():
        print(ch)