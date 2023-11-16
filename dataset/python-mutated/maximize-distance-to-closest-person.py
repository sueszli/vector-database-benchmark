class Solution(object):

    def maxDistToClosest(self, seats):
        if False:
            i = 10
            return i + 15
        '\n        :type seats: List[int]\n        :rtype: int\n        '
        (prev, result) = (-1, 1)
        for i in xrange(len(seats)):
            if seats[i]:
                if prev < 0:
                    result = i
                else:
                    result = max(result, (i - prev) // 2)
                prev = i
        return max(result, len(seats) - 1 - prev)