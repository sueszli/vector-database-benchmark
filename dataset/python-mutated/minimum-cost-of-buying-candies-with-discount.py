class Solution(object):

    def minimumCost(self, cost):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type cost: List[int]\n        :rtype: int\n        '
        cost.sort(reverse=True)
        return sum((x for (i, x) in enumerate(cost) if i % 3 != 2))