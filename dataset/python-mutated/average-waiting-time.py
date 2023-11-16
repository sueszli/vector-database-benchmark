class Solution(object):

    def averageWaitingTime(self, customers):
        if False:
            print('Hello World!')
        '\n        :type customers: List[List[int]]\n        :rtype: float\n        '
        avai = wait = 0.0
        for (a, t) in customers:
            avai = max(avai, a) + t
            wait += avai - a
        return wait / len(customers)