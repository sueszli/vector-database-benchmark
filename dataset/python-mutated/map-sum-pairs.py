import collections

class MapSum(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        '\n        Initialize your data structure here.\n        '
        _trie = lambda : collections.defaultdict(_trie)
        self.__root = _trie()

    def insert(self, key, val):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type key: str\n        :type val: int\n        :rtype: void\n        '
        curr = self.__root
        for c in key:
            curr = curr[c]
        delta = val
        if '_end' in curr:
            delta -= curr['_end']
        curr = self.__root
        for c in key:
            curr = curr[c]
            if '_count' in curr:
                curr['_count'] += delta
            else:
                curr['_count'] = delta
        curr['_end'] = val

    def sum(self, prefix):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type prefix: str\n        :rtype: int\n        '
        curr = self.__root
        for c in prefix:
            if c not in curr:
                return 0
            curr = curr[c]
        return curr['_count']