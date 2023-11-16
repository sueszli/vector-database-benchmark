class Solution(object):

    def minOperationsMaxProfit(self, customers, boardingCost, runningCost):
        if False:
            i = 10
            return i + 15
        '\n        :type customers: List[int]\n        :type boardingCost: int\n        :type runningCost: int\n        :rtype: int\n        '
        max_run = -1
        i = max_prof = prof = waiting = 0
        run = 1
        while i < len(customers) or waiting > 0:
            if i < len(customers):
                waiting += customers[i]
                i += 1
            boarding = min(waiting, 4)
            waiting -= boarding
            prof += boarding * boardingCost - runningCost
            if prof > max_prof:
                max_prof = prof
                max_run = run
            run += 1
        return max_run