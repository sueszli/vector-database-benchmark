import collections

class WordFilter(object):

    def __init__(self, words):
        if False:
            i = 10
            return i + 15
        '\n        :type words: List[str]\n        '
        _trie = lambda : collections.defaultdict(_trie)
        self.__trie = _trie()
        for (weight, word) in enumerate(words):
            word += '#'
            for i in xrange(len(word)):
                cur = self.__trie
                cur['_weight'] = weight
                for j in xrange(i, 2 * len(word) - 1):
                    cur = cur[word[j % len(word)]]
                    cur['_weight'] = weight

    def f(self, prefix, suffix):
        if False:
            while True:
                i = 10
        '\n        :type prefix: str\n        :type suffix: str\n        :rtype: int\n        '
        cur = self.__trie
        for letter in suffix + '#' + prefix:
            if letter not in cur:
                return -1
            cur = cur[letter]
        return cur['_weight']

class Trie(object):

    def __init__(self):
        if False:
            print('Hello World!')
        _trie = lambda : collections.defaultdict(_trie)
        self.__trie = _trie()

    def insert(self, word, i):
        if False:
            i = 10
            return i + 15

        def add_word(cur, i):
            if False:
                i = 10
                return i + 15
            if '_words' not in cur:
                cur['_words'] = []
            cur['_words'].append(i)
        cur = self.__trie
        add_word(cur, i)
        for c in word:
            cur = cur[c]
            add_word(cur, i)

    def find(self, word):
        if False:
            i = 10
            return i + 15
        cur = self.__trie
        for c in word:
            if c not in cur:
                return []
            cur = cur[c]
        return cur['_words']

class WordFilter2(object):

    def __init__(self, words):
        if False:
            return 10
        '\n        :type words: List[str]\n        '
        self.__prefix_trie = Trie()
        self.__suffix_trie = Trie()
        for i in reversed(xrange(len(words))):
            self.__prefix_trie.insert(words[i], i)
            self.__suffix_trie.insert(words[i][::-1], i)

    def f(self, prefix, suffix):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type prefix: str\n        :type suffix: str\n        :rtype: int\n        '
        prefix_match = self.__prefix_trie.find(prefix)
        suffix_match = self.__suffix_trie.find(suffix[::-1])
        (i, j) = (0, 0)
        while i != len(prefix_match) and j != len(suffix_match):
            if prefix_match[i] == suffix_match[j]:
                return prefix_match[i]
            elif prefix_match[i] > suffix_match[j]:
                i += 1
            else:
                j += 1
        return -1