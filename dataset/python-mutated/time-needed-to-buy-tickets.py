class Solution(object):

    def timeRequiredToBuy(self, tickets, k):
        if False:
            print('Hello World!')
        '\n        :type tickets: List[int]\n        :type k: int\n        :rtype: int\n        '
        return sum((min(x, tickets[k] if i <= k else tickets[k] - 1) for (i, x) in enumerate(tickets)))