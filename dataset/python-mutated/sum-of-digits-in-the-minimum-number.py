class Solution(object):

    def sumOfDigits(self, A):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type A: List[int]\n        :rtype: int\n        '
        total = sum([int(c) for c in str(min(A))])
        return 1 if total % 2 == 0 else 0