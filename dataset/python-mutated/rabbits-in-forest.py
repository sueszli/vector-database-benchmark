import collections

class Solution(object):

    def numRabbits(self, answers):
        if False:
            return 10
        '\n        :type answers: List[int]\n        :rtype: int\n        '
        count = collections.Counter(answers)
        return sum(((k + 1 + v - 1) // (k + 1) * (k + 1) for (k, v) in count.iteritems()))