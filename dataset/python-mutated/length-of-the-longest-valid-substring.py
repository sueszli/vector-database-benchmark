import collections

class Solution(object):

    def longestValidSubstring(self, word, forbidden):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type word: str\n        :type forbidden: List[str]\n        :rtype: int\n        '
        _trie = lambda : collections.defaultdict(_trie)
        trie = _trie()
        for w in forbidden:
            reduce(dict.__getitem__, w, trie)['_end']
        result = 0
        right = len(word) - 1
        for left in reversed(xrange(len(word))):
            node = trie
            for i in xrange(left, right + 1):
                if word[i] not in node:
                    break
                node = node[word[i]]
                if '_end' in node:
                    right = i - 1
                    break
            result = max(result, right - left + 1)
        return result