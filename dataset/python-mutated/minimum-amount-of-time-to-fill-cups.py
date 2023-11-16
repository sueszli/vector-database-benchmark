class Solution(object):

    def fillCups(self, amount):
        if False:
            return 10
        '\n        :type amount: List[int]\n        :rtype: int\n        '
        return max(max(amount), (sum(amount) + 1) // 2)

class Solution2(object):

    def fillCups(self, amount):
        if False:
            i = 10
            return i + 15
        '\n        :type amount: List[int]\n        :rtype: int\n        '
        (mx, total) = (max(amount), sum(amount))
        return mx if sum(amount) - mx <= mx else (total + 1) // 2