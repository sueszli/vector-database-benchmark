class Solution(object):

    def findCelebrity(self, n):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :rtype: int\n        '
        candidate = 0
        for i in xrange(1, n):
            if knows(candidate, i):
                candidate = i
        for i in xrange(n):
            candidate_knows_i = knows(candidate, i)
            i_knows_candidate = knows(i, candidate)
            if i != candidate and (candidate_knows_i or not i_knows_candidate):
                return -1
        return candidate