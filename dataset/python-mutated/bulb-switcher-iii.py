class Solution(object):

    def numTimesAllBlue(self, light):
        if False:
            print('Hello World!')
        '\n        :type light: List[int]\n        :rtype: int\n        '
        (result, right) = (0, 0)
        for (i, num) in enumerate(light, 1):
            right = max(right, num)
            result += right == i
        return result