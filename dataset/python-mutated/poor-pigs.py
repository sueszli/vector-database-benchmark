import math

class Solution(object):

    def poorPigs(self, buckets, minutesToDie, minutesToTest):
        if False:
            return 10
        '\n        :type buckets: int\n        :type minutesToDie: int\n        :type minutesToTest: int\n        :rtype: int\n        '
        return int(math.ceil(math.log(buckets) / math.log(minutesToTest / minutesToDie + 1)))