class Solution(object):

    def findMaximalUncoveredRanges(self, n, ranges):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type ranges: List[List[int]]\n        :rtype: List[List[int]]\n        '
        ranges.sort()
        covered = [[-1, -1]]
        for (left, right) in ranges:
            if covered[-1][1] < left:
                covered.append([left, right])
                continue
            covered[-1][1] = max(covered[-1][1], right)
        covered.append([n, n])
        return [[covered[i - 1][1] + 1, covered[i][0] - 1] for i in xrange(1, len(covered)) if covered[i - 1][1] + 1 <= covered[i][0] - 1]