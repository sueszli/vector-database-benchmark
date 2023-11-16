import random
import bisect

class Solution(object):

    def __init__(self, w):
        if False:
            i = 10
            return i + 15
        '\n        :type w: List[int]\n        '
        self.__prefix_sum = list(w)
        for i in xrange(1, len(w)):
            self.__prefix_sum[i] += self.__prefix_sum[i - 1]

    def pickIndex(self):
        if False:
            i = 10
            return i + 15
        '\n        :rtype: int\n        '
        target = random.randint(0, self.__prefix_sum[-1] - 1)
        return bisect.bisect_right(self.__prefix_sum, target)