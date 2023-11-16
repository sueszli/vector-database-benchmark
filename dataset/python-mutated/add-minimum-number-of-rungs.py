class Solution(object):

    def addRungs(self, rungs, dist):
        if False:
            print('Hello World!')
        '\n        :type rungs: List[int]\n        :type dist: int\n        :rtype: int\n        '

        def ceil_divide(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return (a + (b - 1)) // b
        result = prev = 0
        for curr in rungs:
            result += ceil_divide(curr - prev, dist) - 1
            prev = curr
        return result