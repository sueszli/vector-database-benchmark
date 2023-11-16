import itertools

class Solution(object):

    def canConvertString(self, s, t, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type s: str\n        :type t: str\n        :type k: int\n        :rtype: bool\n        '
        if len(s) != len(t):
            return False
        cnt = [0] * 26
        for (a, b) in itertools.izip(s, t):
            diff = (ord(b) - ord(a)) % len(cnt)
            if diff != 0 and cnt[diff] * len(cnt) + diff > k:
                return False
            cnt[diff] += 1
        return True