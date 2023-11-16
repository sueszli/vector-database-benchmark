"""
Topic: 代理迭代
Desc : 
"""

class Node:

    def __init__(self, value):
        if False:
            for i in range(10):
                print('nop')
        self._value = value
        self._children = []

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'Node({!r})'.format(self._value)

    def add_child(self, node):
        if False:
            print('Hello World!')
        self._children.append(node)

    def __iter__(self):
        if False:
            return 10
        return iter(self._children)
if __name__ == '__main__':
    root = Node(0)
    child1 = Node(1)
    child2 = Node(2)
    root.add_child(child1)
    root.add_child(child2)
    for ch in root:
        print(ch)