class Solution(object):

    def maximumNumberOfOnes(self, width, height, sideLength, maxOnes):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type width: int\n        :type height: int\n        :type sideLength: int\n        :type maxOnes: int\n        :rtype: int\n        '
        if width < height:
            (width, height) = (height, width)
        (R, r) = divmod(height, sideLength)
        (C, c) = divmod(width, sideLength)
        assert R <= C
        area_counts = [(r * c, (R + 1) * (C + 1)), (r * (sideLength - c), (R + 1) * C), ((sideLength - r) * c, R * (C + 1)), ((sideLength - r) * (sideLength - c), R * C)]
        result = 0
        for (area, count) in area_counts:
            area = min(maxOnes, area)
            result += count * area
            maxOnes -= area
            if not maxOnes:
                break
        return result