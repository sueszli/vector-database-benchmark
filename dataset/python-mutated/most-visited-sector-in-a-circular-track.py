class Solution(object):

    def mostVisited(self, n, rounds):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type rounds: List[int]\n        :rtype: List[int]\n        '
        return range(rounds[0], rounds[-1] + 1) or range(1, rounds[-1] + 1) + range(rounds[0], n + 1)