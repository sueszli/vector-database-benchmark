import collections

class Solution(object):

    def minimumRounds(self, tasks):
        if False:
            i = 10
            return i + 15
        '\n        :type tasks: List[int]\n        :rtype: int\n        '
        cnt = collections.Counter(tasks)
        return sum(((x + 2) // 3 for x in cnt.itervalues())) if 1 not in cnt.itervalues() else -1