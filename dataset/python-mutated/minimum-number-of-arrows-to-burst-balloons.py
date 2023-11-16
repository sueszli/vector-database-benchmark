class Solution(object):

    def findMinArrowShots(self, points):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type points: List[List[int]]\n        :rtype: int\n        '
        if not points:
            return 0
        points.sort()
        result = 0
        i = 0
        while i < len(points):
            j = i + 1
            right_bound = points[i][1]
            while j < len(points) and points[j][0] <= right_bound:
                right_bound = min(right_bound, points[j][1])
                j += 1
            result += 1
            i = j
        return result