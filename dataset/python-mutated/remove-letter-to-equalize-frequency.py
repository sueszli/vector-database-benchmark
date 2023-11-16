import collections

class Solution(object):

    def equalFrequency(self, word):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type word: str\n        :rtype: bool\n        '
        cnt = collections.Counter(collections.Counter(word).itervalues())
        if len(cnt) > 2:
            return False
        if len(cnt) == 1:
            a = cnt.keys()[0]
            return a == 1 or cnt[a] == 1
        (a, b) = cnt.keys()
        if a > b:
            (a, b) = (b, a)
        return a == 1 and cnt[a] == 1 or (a + 1 == b and cnt[b] == 1)
import collections

class Solution2(object):

    def equalFrequency(self, word):
        if False:
            while True:
                i = 10
        '\n        :type word: str\n        :rtype: bool\n        '
        cnt = collections.Counter(collections.Counter(word))
        for c in word:
            cnt[c] -= 1
            if len(collections.Counter((c for c in cnt.itervalues() if c))) == 1:
                return True
            cnt[c] += 1
        return False