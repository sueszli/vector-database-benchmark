class Solution(object):

    def grayCode(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: List[int]\n        '
        result = [0]
        for i in xrange(n):
            for n in reversed(result):
                result.append(1 << i | n)
        return result

class Solution2(object):

    def grayCode(self, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :rtype: List[int]\n        '
        return [i >> 1 ^ i for i in xrange(1 << n)]