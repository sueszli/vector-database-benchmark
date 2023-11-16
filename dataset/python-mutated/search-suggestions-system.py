import collections

class TrieNode(object):

    def __init__(self):
        if False:
            print('Hello World!')
        self.__TOP_COUNT = 3
        self.leaves = collections.defaultdict(TrieNode)
        self.infos = []

    def insert(self, words, i):
        if False:
            return 10
        curr = self
        for c in words[i]:
            curr = curr.leaves[c]
            curr.add_info(words, i)

    def add_info(self, words, i):
        if False:
            print('Hello World!')
        self.infos.append(i)
        self.infos.sort(key=lambda x: words[x])
        if len(self.infos) > self.__TOP_COUNT:
            self.infos.pop()

class Solution(object):

    def suggestedProducts(self, products, searchWord):
        if False:
            return 10
        '\n        :type products: List[str]\n        :type searchWord: str\n        :rtype: List[List[str]]\n        '
        trie = TrieNode()
        for i in xrange(len(products)):
            trie.insert(products, i)
        result = [[] for _ in xrange(len(searchWord))]
        for (i, c) in enumerate(searchWord):
            if c not in trie.leaves:
                break
            trie = trie.leaves[c]
            result[i] = map(lambda x: products[x], trie.infos)
        return result

class TrieNode2(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__TOP_COUNT = 3
        self.leaves = collections.defaultdict(TrieNode2)
        self.infos = []

    def insert(self, words, i):
        if False:
            print('Hello World!')
        curr = self
        for c in words[i]:
            curr = curr.leaves[c]
            curr.add_info(i)

    def add_info(self, i):
        if False:
            while True:
                i = 10
        if len(self.infos) == self.__TOP_COUNT:
            return
        self.infos.append(i)

class Solution2(object):

    def suggestedProducts(self, products, searchWord):
        if False:
            print('Hello World!')
        '\n        :type products: List[str]\n        :type searchWord: str\n        :rtype: List[List[str]]\n        '
        products.sort()
        trie = TrieNode2()
        for i in xrange(len(products)):
            trie.insert(products, i)
        result = [[] for _ in xrange(len(searchWord))]
        for (i, c) in enumerate(searchWord):
            if c not in trie.leaves:
                break
            trie = trie.leaves[c]
            result[i] = map(lambda x: products[x], trie.infos)
        return result
import bisect

class Solution3(object):

    def suggestedProducts(self, products, searchWord):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type products: List[str]\n        :type searchWord: str\n        :rtype: List[List[str]]\n        '
        products.sort()
        result = []
        prefix = ''
        for (i, c) in enumerate(searchWord):
            prefix += c
            start = bisect.bisect_left(products, prefix)
            new_products = []
            for j in xrange(start, len(products)):
                if not (i < len(products[j]) and products[j][i] == c):
                    break
                new_products.append(products[j])
            products = new_products
            result.append(products[:3])
        return result