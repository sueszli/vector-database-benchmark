class Solution(object):

    def countDistinctIntegers(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def reverse(n):
            if False:
                i = 10
                return i + 15
            result = 0
            while n:
                result = result * 10 + n % 10
                n //= 10
            return result
        return len({y for x in nums for y in (x, reverse(x))})

class Solution2(object):

    def countDistinctIntegers(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        return len({y for x in nums for y in (x, int(str(x)[::-1]))})