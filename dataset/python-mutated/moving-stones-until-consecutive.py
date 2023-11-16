class Solution(object):

    def numMovesStones(self, a, b, c):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type a: int\n        :type b: int\n        :type c: int\n        :rtype: List[int]\n        '
        s = [a, b, c]
        s.sort()
        if s[0] + 1 == s[1] and s[1] + 1 == s[2]:
            return [0, 0]
        return [1 if s[0] + 2 >= s[1] or s[1] + 2 >= s[2] else 2, s[2] - s[0] - 2]

class Solution2(object):

    def numMovesStones(self, a, b, c):
        if False:
            i = 10
            return i + 15
        '\n        :type a: int\n        :type b: int\n        :type c: int\n        :rtype: List[int]\n        '
        stones = [a, b, c]
        stones.sort()
        (left, min_moves) = (0, float('inf'))
        max_moves = stones[-1] - stones[0] - (len(stones) - 1)
        for right in xrange(len(stones)):
            while stones[right] - stones[left] + 1 > len(stones):
                left += 1
            min_moves = min(min_moves, len(stones) - (right - left + 1))
        return [min_moves, max_moves]