import collections
import functools

class Solution(object):

    def addBoldTag(self, s, dict):
        if False:
            return 10
        '\n        :type s: str\n        :type dict: List[str]\n        :rtype: str\n        '
        lookup = [0] * len(s)
        for d in dict:
            pos = s.find(d)
            while pos != -1:
                lookup[pos:pos + len(d)] = [1] * len(d)
                pos = s.find(d, pos + 1)
        result = []
        for i in xrange(len(s)):
            if lookup[i] and (i == 0 or not lookup[i - 1]):
                result.append('<b>')
            result.append(s[i])
            if lookup[i] and (i == len(s) - 1 or not lookup[i + 1]):
                result.append('</b>')
        return ''.join(result)

class Solution2(object):

    def addBoldTag(self, s, words):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type words: List[str]\n        :rtype: str\n        '
        _trie = lambda : collections.defaultdict(_trie)
        trie = _trie()
        for (i, word) in enumerate(words):
            functools.reduce(dict.__getitem__, word, trie).setdefault('_end')
        lookup = [False] * len(s)
        for i in xrange(len(s)):
            curr = trie
            k = -1
            for j in xrange(i, len(s)):
                if s[j] not in curr:
                    break
                curr = curr[s[j]]
                if '_end' in curr:
                    k = j
            for j in xrange(i, k + 1):
                lookup[j] = True
        result = []
        for i in xrange(len(s)):
            if lookup[i] and (i == 0 or not lookup[i - 1]):
                result.append('<b>')
            result.append(s[i])
            if lookup[i] and (i == len(s) - 1 or not lookup[i + 1]):
                result.append('</b>')
        return ''.join(result)