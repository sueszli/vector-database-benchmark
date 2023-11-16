import collections

class Solution(object):

    def makeEqual(self, words):
        if False:
            return 10
        '\n        :type words: List[str]\n        :rtype: bool\n        '
        cnt = collections.defaultdict(int)
        for w in words:
            for c in w:
                cnt[c] += 1
        return all((v % len(words) == 0 for v in cnt.itervalues()))