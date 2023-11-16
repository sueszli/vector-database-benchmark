class Solution(object):

    def isBoomerang(self, points):
        if False:
            print('Hello World!')
        '\n        :type points: List[List[int]]\n        :rtype: bool\n        '
        return (points[0][0] - points[1][0]) * (points[0][1] - points[2][1]) - (points[0][0] - points[2][0]) * (points[0][1] - points[1][1]) != 0