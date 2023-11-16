class Solution(object):

    def canEat(self, candiesCount, queries):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type candiesCount: List[int]\n        :type queries: List[List[int]]\n        :rtype: List[bool]\n        '
        prefix = [0] * (len(candiesCount) + 1)
        for (i, c) in enumerate(candiesCount):
            prefix[i + 1] = prefix[i] + c
        return [prefix[t] // c < d + 1 <= prefix[t + 1] // 1 for (t, d, c) in queries]