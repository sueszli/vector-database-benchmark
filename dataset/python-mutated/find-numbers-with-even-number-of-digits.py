import bisect

class Solution(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        M = 10 ** 5
        self.__lookup = [0]
        i = 10
        while i < M:
            self.__lookup.append(i)
            i *= 10
        self.__lookup.append(i)

    def findNumbers(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def digit_count(n):
            if False:
                return 10
            return bisect.bisect_right(self.__lookup, n)
        return sum((digit_count(n) % 2 == 0 for n in nums))

class Solution2(object):

    def findNumbers(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def digit_count(n):
            if False:
                for i in range(10):
                    print('nop')
            result = 0
            while n:
                n //= 10
                result += 1
            return result
        return sum((digit_count(n) % 2 == 0 for n in nums))

class Solution3(object):

    def findNumbers(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        return sum((len(str(n)) % 2 == 0 for n in nums))