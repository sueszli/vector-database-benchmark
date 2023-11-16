import collections

class Solution(object):

    def sumPrefixScores(self, words):
        if False:
            return 10
        '\n        :type words: List[str]\n        :rtype: List[int]\n        '
        _trie = lambda : collections.defaultdict(_trie)
        trie = _trie()
        for w in words:
            curr = trie
            for c in w:
                curr = curr[c]
                curr['_cnt'] = curr['_cnt'] + 1 if '_cnt' in curr else 1
        result = []
        for w in words:
            cnt = 0
            curr = trie
            for c in w:
                curr = curr[c]
                cnt += curr['_cnt']
            result.append(cnt)
        return result