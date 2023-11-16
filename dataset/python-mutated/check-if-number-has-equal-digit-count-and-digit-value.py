import collections

class Solution(object):

    def digitCount(self, num):
        if False:
            print('Hello World!')
        '\n        :type num: str\n        :rtype: bool\n        '
        cnt = collections.Counter(num)
        return all((cnt[str(i)] == int(x) for (i, x) in enumerate(num)))