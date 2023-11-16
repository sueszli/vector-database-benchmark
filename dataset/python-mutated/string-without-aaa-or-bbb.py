class Solution(object):

    def strWithout3a3b(self, A, B):
        if False:
            i = 10
            return i + 15
        '\n        :type A: int\n        :type B: int\n        :rtype: str\n        '
        result = []
        put_A = None
        while A or B:
            if len(result) >= 2 and result[-1] == result[-2]:
                put_A = result[-1] == 'b'
            else:
                put_A = A >= B
            if put_A:
                A -= 1
                result.append('a')
            else:
                B -= 1
                result.append('b')
        return ''.join(result)