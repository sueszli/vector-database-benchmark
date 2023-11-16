import collections

class Solution(object):

    def taskSchedulerII(self, tasks, space):
        if False:
            while True:
                i = 10
        '\n        :type tasks: List[int]\n        :type space: int\n        :rtype: int\n        '
        lookup = collections.defaultdict(int)
        result = 0
        for t in tasks:
            result = max(lookup[t], result + 1)
            lookup[t] = result + space + 1
        return result