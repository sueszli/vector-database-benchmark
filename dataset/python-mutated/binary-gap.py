class Solution(object):

    def binaryGap(self, N):
        if False:
            print('Hello World!')
        '\n        :type N: int\n        :rtype: int\n        '
        result = 0
        last = None
        for i in xrange(32):
            if N >> i & 1:
                if last is not None:
                    result = max(result, i - last)
                last = i
        return result