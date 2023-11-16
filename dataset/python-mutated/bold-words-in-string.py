import collections
import functools

class Solution(object):

    def boldWords(self, words, S):
        if False:
            print('Hello World!')
        '\n        :type words: List[str]\n        :type S: str\n        :rtype: str\n        '
        _trie = lambda : collections.defaultdict(_trie)
        trie = _trie()
        for (i, word) in enumerate(words):
            functools.reduce(dict.__getitem__, word, trie).setdefault('_end')
        lookup = [False] * len(S)
        for i in xrange(len(S)):
            curr = trie
            k = -1
            for j in xrange(i, len(S)):
                if S[j] not in curr:
                    break
                curr = curr[S[j]]
                if '_end' in curr:
                    k = j
            for j in xrange(i, k + 1):
                lookup[j] = True
        result = []
        for i in xrange(len(S)):
            if lookup[i] and (i == 0 or not lookup[i - 1]):
                result.append('<b>')
            result.append(S[i])
            if lookup[i] and (i == len(S) - 1 or not lookup[i + 1]):
                result.append('</b>')
        return ''.join(result)

class Solution2(object):

    def boldWords(self, words, S):
        if False:
            print('Hello World!')
        '\n        :type words: List[str]\n        :type S: str\n        :rtype: str\n        '
        lookup = [0] * len(S)
        for d in words:
            pos = S.find(d)
            while pos != -1:
                lookup[pos:pos + len(d)] = [1] * len(d)
                pos = S.find(d, pos + 1)
        result = []
        for i in xrange(len(S)):
            if lookup[i] and (i == 0 or not lookup[i - 1]):
                result.append('<b>')
            result.append(S[i])
            if lookup[i] and (i == len(S) - 1 or not lookup[i + 1]):
                result.append('</b>')
        return ''.join(result)