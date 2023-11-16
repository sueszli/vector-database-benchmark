"""
Topic: 弱引用
Desc : 
"""
import weakref

class Node:

    def __init__(self, value):
        if False:
            while True:
                i = 10
        self.value = value
        self._parent = None
        self.children = []

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'Node({!r:})'.format(self.value)

    @property
    def parent(self):
        if False:
            return 10
        return None if self._parent is None else self._parent()

    @parent.setter
    def parent(self, node):
        if False:
            print('Hello World!')
        self._parent = weakref.ref(node)

    def add_child(self, child):
        if False:
            while True:
                i = 10
        self.children.append(child)
        child.parent = self

class Data:

    def __del__(self):
        if False:
            return 10
        print('Data.__del__')

class Node1:

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.data = Data()
        self.parent = None
        self.children = []

    def add_child(self, child):
        if False:
            for i in range(10):
                print('nop')
        self.children.append(child)
        child.parent = self
a = Data()
del a
a = Node1()
del a
a = Node1()
a.add_child(Node1())
print('--------last del start------------')
del a
print('--------last del end------------')
print('11111111111111111')