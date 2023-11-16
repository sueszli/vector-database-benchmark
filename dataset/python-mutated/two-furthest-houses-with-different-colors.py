class Solution(object):

    def maxDistance(self, colors):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type colors: List[int]\n        :rtype: int\n        '
        result = 0
        for (i, x) in enumerate(colors):
            if x != colors[0]:
                result = max(result, i)
            if x != colors[-1]:
                result = max(result, len(colors) - 1 - i)
        return result