class Solution(object):

    def minimumNumbers(self, num, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type num: int\n        :type k: int\n        :rtype: int\n        '
        return next((i for i in xrange(1, (min(num // k, 10) if k else 1) + 1) if (num - i * k) % 10 == 0), -1) if num else 0