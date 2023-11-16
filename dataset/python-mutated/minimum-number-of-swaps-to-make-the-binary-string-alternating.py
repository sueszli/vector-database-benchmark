class Solution(object):

    def minSwaps(self, s):
        if False:
            while True:
                i = 10
        '\n        :type s: str\n        :rtype: int\n        '

        def cost(s, x):
            if False:
                for i in range(10):
                    print('nop')
            diff = 0
            for c in s:
                diff += int(c) != x
                x ^= 1
            return diff // 2
        ones = s.count('1')
        zeros = len(s) - ones
        if abs(ones - zeros) > 1:
            return -1
        if ones > zeros:
            return cost(s, 1)
        if ones < zeros:
            return cost(s, 0)
        return min(cost(s, 1), cost(s, 0))