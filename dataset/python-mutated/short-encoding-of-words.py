import collections
import functools

class Solution(object):

    def minimumLengthEncoding(self, words):
        if False:
            print('Hello World!')
        '\n        :type words: List[str]\n        :rtype: int\n        '
        words = list(set(words))
        _trie = lambda : collections.defaultdict(_trie)
        trie = _trie()
        nodes = [functools.reduce(dict.__getitem__, word[::-1], trie) for word in words]
        return sum((len(word) + 1 for (i, word) in enumerate(words) if len(nodes[i]) == 0))