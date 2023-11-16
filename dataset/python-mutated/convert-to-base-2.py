class Solution(object):

    def baseNeg2(self, N):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type N: int\n        :rtype: str\n        '
        result = []
        while N:
            result.append(str(-N & 1))
            N = -(N >> 1)
        result.reverse()
        return ''.join(result) if result else '0'

class Solution2(object):

    def baseNeg2(self, N):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type N: int\n        :rtype: str\n        '
        BASE = -2
        result = []
        while N:
            (N, r) = divmod(N, BASE)
            if r < 0:
                r -= BASE
                N += 1
            result.append(str(r))
        result.reverse()
        return ''.join(result) if result else '0'