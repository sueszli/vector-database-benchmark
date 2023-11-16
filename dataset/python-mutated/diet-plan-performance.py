import itertools

class Solution(object):

    def dietPlanPerformance(self, calories, k, lower, upper):
        if False:
            return 10
        '\n        :type calories: List[int]\n        :type k: int\n        :type lower: int\n        :type upper: int\n        :rtype: int\n        '
        total = sum(itertools.islice(calories, 0, k))
        result = int(total > upper) - int(total < lower)
        for i in xrange(k, len(calories)):
            total += calories[i] - calories[i - k]
            result += int(total > upper) - int(total < lower)
        return result