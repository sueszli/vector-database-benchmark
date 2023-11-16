class Trie(object):
    """
    A trie (prefix tree) where the keys are sequences of hashable objects.

    It behaves similarly to a dictionary, except that the keys can be lists or
    other sequences.

    Examples:
        >>> from featuretools.utils import Trie
        >>> trie = Trie(default=str)
        >>> # Set a value
        >>> trie.get_node([1, 2, 3]).value = '123'
        >>> # Get a value
        >>> trie.get_node([1, 2, 3]).value
        '123'
        >>> # Overwrite a value
        >>> trie.get_node([1, 2, 3]).value = 'updated'
        >>> trie.get_node([1, 2, 3]).value
        'updated'
        >>> # Getting a key that has not been set returns the default value.
        >>> trie.get_node([1, 2]).value
        ''
    """

    def __init__(self, default=lambda : None, path_constructor=list):
        if False:
            print('Hello World!')
        '\n        default: A function returning the value to use for new nodes.\n        path_constructor: A function which constructs a path from a list. The\n            path type must support addition (concatenation).\n        '
        self.value = default()
        self._children = {}
        self._default = default
        self._path_constructor = path_constructor

    def children(self):
        if False:
            print('Hello World!')
        "\n        A list of pairs of the edges from this node and the nodes they point\n        to.\n\n        Examples:\n            >>> from featuretools.utils import Trie\n            >>> trie = Trie(default=str)\n            >>> trie.get_node([1, 2]).value = '12'\n            >>> trie.get_node([3]).value = '3'\n            >>> children = trie.children()\n            >>> first_edge, first_child = children[0]\n            >>> first_edge\n            1\n            >>> first_child.value\n            ''\n            >>> second_edge, second_child = children[1]\n            >>> second_edge\n            3\n            >>> second_child.value\n            '3'\n        "
        return list(self._children.items())

    def get_node(self, path):
        if False:
            print('Hello World!')
        "\n        Get the sub-trie at the given path. If it does not yet exist initialize\n        it with the default value.\n\n        Examples:\n            >>> from featuretools.utils import Trie\n            >>> t = Trie()\n            >>> t.get_node([1, 2, 3]).value = '123'\n            >>> t.get_node([1, 2, 4]).value = '124'\n            >>> sub = t.get_node([1, 2])\n            >>> sub.get_node([3]).value\n            '123'\n            >>> sub.get_node([4]).value\n            '124'\n        "
        if path:
            first = path[0]
            rest = path[1:]
            if first in self._children:
                sub_trie = self._children[first]
            else:
                sub_trie = Trie(default=self._default, path_constructor=self._path_constructor)
                self._children[first] = sub_trie
            return sub_trie.get_node(rest)
        else:
            return self

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Iterate over all values in the trie. Yields tuples of (path, value).\n\n        Implemented using depth first search.\n        '
        yield (self._path_constructor([]), self.value)
        for (key, sub_trie) in self.children():
            path_to_children = self._path_constructor([key])
            for (sub_path, value) in sub_trie:
                path = path_to_children + sub_path
                yield (path, value)