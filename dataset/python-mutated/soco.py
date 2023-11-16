"""
The SCEngine class uses the ``sortedcontainers`` package to implement an
Index engine for Tables.
"""
from collections import OrderedDict
from itertools import starmap
from astropy.utils.compat.optional_deps import HAS_SORTEDCONTAINERS
if HAS_SORTEDCONTAINERS:
    from sortedcontainers import SortedList

class Node:
    __slots__ = ('key', 'value')

    def __init__(self, key, value):
        if False:
            print('Hello World!')
        self.key = key
        self.value = value

    def __lt__(self, other):
        if False:
            print('Hello World!')
        if other.__class__ is Node:
            return (self.key, self.value) < (other.key, other.value)
        return self.key < other

    def __le__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if other.__class__ is Node:
            return (self.key, self.value) <= (other.key, other.value)
        return self.key <= other

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if other.__class__ is Node:
            return (self.key, self.value) == (other.key, other.value)
        return self.key == other

    def __ne__(self, other):
        if False:
            print('Hello World!')
        if other.__class__ is Node:
            return (self.key, self.value) != (other.key, other.value)
        return self.key != other

    def __gt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if other.__class__ is Node:
            return (self.key, self.value) > (other.key, other.value)
        return self.key > other

    def __ge__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if other.__class__ is Node:
            return (self.key, self.value) >= (other.key, other.value)
        return self.key >= other
    __hash__ = None

    def __repr__(self):
        if False:
            while True:
                i = 10
        return f'Node({self.key!r}, {self.value!r})'

class SCEngine:
    """
    Fast tree-based implementation for indexing, using the
    ``sortedcontainers`` package.

    Parameters
    ----------
    data : Table
        Sorted columns of the original table
    row_index : Column object
        Row numbers corresponding to data columns
    unique : bool
        Whether the values of the index must be unique.
        Defaults to False.
    """

    def __init__(self, data, row_index, unique=False):
        if False:
            i = 10
            return i + 15
        if not HAS_SORTEDCONTAINERS:
            raise ImportError('sortedcontainers is needed for using SCEngine')
        node_keys = map(tuple, data)
        self._nodes = SortedList(starmap(Node, zip(node_keys, row_index)))
        self._unique = unique

    def add(self, key, value):
        if False:
            return 10
        '\n        Add a key, value pair.\n        '
        if self._unique and key in self._nodes:
            message = f'duplicate {key!r} in unique index'
            raise ValueError(message)
        self._nodes.add(Node(key, value))

    def find(self, key):
        if False:
            return 10
        '\n        Find rows corresponding to the given key.\n        '
        return [node.value for node in self._nodes.irange(key, key)]

    def remove(self, key, data=None):
        if False:
            while True:
                i = 10
        '\n        Remove data from the given key.\n        '
        if data is not None:
            item = Node(key, data)
            try:
                self._nodes.remove(item)
            except ValueError:
                return False
            return True
        items = list(self._nodes.irange(key, key))
        for item in items:
            self._nodes.remove(item)
        return bool(items)

    def shift_left(self, row):
        if False:
            return 10
        '\n        Decrement rows larger than the given row.\n        '
        for node in self._nodes:
            if node.value > row:
                node.value -= 1

    def shift_right(self, row):
        if False:
            i = 10
            return i + 15
        '\n        Increment rows greater than or equal to the given row.\n        '
        for node in self._nodes:
            if node.value >= row:
                node.value += 1

    def items(self):
        if False:
            print('Hello World!')
        '\n        Return a list of key, data tuples.\n        '
        result = OrderedDict()
        for node in self._nodes:
            if node.key in result:
                result[node.key].append(node.value)
            else:
                result[node.key] = [node.value]
        return result.items()

    def sort(self):
        if False:
            print('Hello World!')
        '\n        Make row order align with key order.\n        '
        for (index, node) in enumerate(self._nodes):
            node.value = index

    def sorted_data(self):
        if False:
            return 10
        '\n        Return a list of rows in order sorted by key.\n        '
        return [node.value for node in self._nodes]

    def range(self, lower, upper, bounds=(True, True)):
        if False:
            while True:
                i = 10
        '\n        Return row values in the given range.\n        '
        iterator = self._nodes.irange(lower, upper, bounds)
        return [node.value for node in iterator]

    def replace_rows(self, row_map):
        if False:
            i = 10
            return i + 15
        '\n        Replace rows with the values in row_map.\n        '
        nodes = [node for node in self._nodes if node.value in row_map]
        for node in nodes:
            node.value = row_map[node.value]
        self._nodes.clear()
        self._nodes.update(nodes)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        if len(self._nodes) > 6:
            nodes = list(self._nodes[:3]) + ['...'] + list(self._nodes[-3:])
        else:
            nodes = self._nodes
        nodes_str = ', '.join((str(node) for node in nodes))
        return f'<{self.__class__.__name__} nodes={nodes_str}>'