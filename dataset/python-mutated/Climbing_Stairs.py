class Solution:

    def __init__(self):
        if False:
            print('Hello World!')
        self.ways = [0] * 46

    def climbStairs(self, n: int) -> int:
        if False:
            i = 10
            return i + 15
        if n <= 2:
            return n
        elif self.ways[n] != 0:
            return self.ways[n]
        else:
            self.ways[n] = self.climbStairs(n - 1) + self.climbStairs(n - 2)
            return self.ways[n]