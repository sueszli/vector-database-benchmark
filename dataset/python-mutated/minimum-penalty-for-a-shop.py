class Solution(object):

    def bestClosingTime(self, customers):
        if False:
            i = 10
            return i + 15
        '\n        :type customers: str\n        :rtype: int\n        '
        result = mx = curr = 0
        for (i, x) in enumerate(customers):
            curr += 1 if x == 'Y' else -1
            if curr > mx:
                mx = curr
                result = i + 1
        return result