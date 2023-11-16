import collections

class Solution(object):

    def countWords(self, words1, words2):
        if False:
            print('Hello World!')
        '\n        :type words1: List[str]\n        :type words2: List[str]\n        :rtype: int\n        '
        cnt = collections.Counter(words1)
        for c in words2:
            if cnt[c] < 2:
                cnt[c] -= 1
        return sum((v == 0 for v in cnt.itervalues()))