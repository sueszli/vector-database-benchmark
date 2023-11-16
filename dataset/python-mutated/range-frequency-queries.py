import collections
import bisect

class RangeFreqQuery(object):

    def __init__(self, arr):
        if False:
            print('Hello World!')
        '\n        :type arr: List[int]\n        '
        self.__idxs = collections.defaultdict(list)
        for (i, x) in enumerate(arr):
            self.__idxs[x].append(i)

    def query(self, left, right, value):
        if False:
            while True:
                i = 10
        '\n        :type left: int\n        :type right: int\n        :type value: int\n        :rtype: int\n        '
        return bisect.bisect_right(self.__idxs[value], right) - bisect.bisect_left(self.__idxs[value], left)