class Solution(object):

    def minimumMoney(self, transactions):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type transactions: List[List[int]]\n        :rtype: int\n        '
        return sum((max(a - b, 0) for (a, b) in transactions)) + max((a - max(a - b, 0) for (a, b) in transactions))