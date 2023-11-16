class Solution(object):

    def lastStoneWeightII(self, stones):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type stones: List[int]\n        :rtype: int\n        '
        dp = {0}
        for stone in stones:
            dp |= {stone + i for i in dp}
        S = sum(stones)
        return min((abs(i - (S - i)) for i in dp))