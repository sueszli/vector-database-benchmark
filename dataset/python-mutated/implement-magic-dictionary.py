import collections

class MagicDictionary(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        '\n        Initialize your data structure here.\n        '
        _trie = lambda : collections.defaultdict(_trie)
        self.trie = _trie()

    def buildDict(self, dictionary):
        if False:
            return 10
        '\n        Build a dictionary through a list of words\n        :type dictionary: List[str]\n        :rtype: void\n        '
        for word in dictionary:
            reduce(dict.__getitem__, word, self.trie).setdefault('_end')

    def search(self, word):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns if there is any word in the trie that equals to the given word after modifying exactly one character\n        :type word: str\n        :rtype: bool\n        '

        def find(word, curr, i, mistakeAllowed):
            if False:
                i = 10
                return i + 15
            if i == len(word):
                return '_end' in curr and (not mistakeAllowed)
            if word[i] not in curr:
                return any((find(word, curr[c], i + 1, False) for c in curr if c != '_end')) if mistakeAllowed else False
            if mistakeAllowed:
                return find(word, curr[word[i]], i + 1, True) or any((find(word, curr[c], i + 1, False) for c in curr if c not in ('_end', word[i])))
            return find(word, curr[word[i]], i + 1, False)
        return find(word, self.trie, 0, True)