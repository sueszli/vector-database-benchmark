class Solution(object):

    def sortByBits(self, arr):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arr: List[int]\n        :rtype: List[int]\n        '

        def popcount(n):
            if False:
                print('Hello World!')
            result = 0
            while n:
                n &= n - 1
                result += 1
            return result
        arr.sort(key=lambda x: (popcount(x), x))
        return arr