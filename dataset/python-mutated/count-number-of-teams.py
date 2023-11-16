class Solution(object):

    def numTeams(self, rating):
        if False:
            while True:
                i = 10
        '\n        :type rating: List[int]\n        :rtype: int\n        '
        result = 0
        for i in xrange(1, len(rating) - 1):
            (less, greater) = ([0] * 2, [0] * 2)
            for j in xrange(len(rating)):
                if rating[i] > rating[j]:
                    less[i < j] += 1
                if rating[i] < rating[j]:
                    greater[i < j] += 1
            result += less[0] * greater[1] + greater[0] * less[1]
        return result