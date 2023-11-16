"""
This class represents a trie[str, typing.Any]
"""
import typing

class Trie:
    """
    This class represents a trie[str, typing.Any]
    """

    class TrieNode:
        """
        This class represents a node in a trie
        """

        def __init__(self, value: typing.Optional[typing.Any]=None):
            if False:
                return 10
            self._children: typing.Dict[str, 'Trie.TrieNode'] = {}
            self._value: typing.Optional[typing.Any] = value

        def __len__(self) -> int:
            if False:
                for i in range(10):
                    print('nop')
            return (0 if self._value is None else 1) + sum([len(v) for (k, v) in self._children.items()])

        def get_value(self) -> typing.Any:
            if False:
                for i in range(10):
                    print('nop')
            '\n            This function returns the value of this TrieNode\n            :return:    the value of this TrieNode\n            '
            return self._value

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._root: typing.Optional[Trie.TrieNode] = None

    def __getitem__(self, item) -> typing.Optional[typing.Any]:
        if False:
            return 10
        n: typing.Optional[Trie.TrieNode] = self._root
        if n is None:
            return None
        for c in item:
            if c in n._children:
                n = n._children[c]
            else:
                return None
        assert n is not None, 'unexpected error while performing __getitem__ on Trie'
        return n.get_value()

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        return 0 if self._root is None else len(self._root)

    def __setitem__(self, key, value):
        if False:
            print('Hello World!')
        n: typing.Optional[Trie.TrieNode] = self._root
        if n is None:
            self._root = Trie.TrieNode()
            n = self._root
        assert n is not None, 'unexpected error while performing __setitem__ on Trie'
        for c in key:
            if c not in n._children:
                n._children[c] = Trie.TrieNode()
            n = n._children[c]
        assert n is not None, 'unexpected error while performing __setitem__ on Trie'
        n._value = value
        return self