from collections import defaultdict
from operator import getitem

class Solution(object):

    def longestWord(self, words):
        if False:
            while True:
                i = 10
        '\n        :type words: List[str]\n        :rtype: str\n        '
        _trie = lambda : defaultdict(_trie)
        trie = _trie()
        for (i, word) in enumerate(words):
            reduce(getitem, word, trie)['_end'] = i
        stack = trie.values()
        result = ''
        while stack:
            curr = stack.pop()
            if '_end' in curr:
                word = words[curr['_end']]
                if len(word) > len(result) or (len(word) == len(result) and word < result):
                    result = word
                stack += [curr[letter] for letter in curr if letter != '_end']
        return result