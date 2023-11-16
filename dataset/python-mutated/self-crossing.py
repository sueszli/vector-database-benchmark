class Solution(object):

    def isSelfCrossing(self, x):
        if False:
            return 10
        '\n        :type x: List[int]\n        :rtype: bool\n        '
        if len(x) >= 5 and x[3] == x[1] and (x[4] + x[0] >= x[2]):
            return True
        for i in xrange(3, len(x)):
            if x[i] >= x[i - 2] and x[i - 3] >= x[i - 1]:
                return True
            elif i >= 5 and x[i - 4] <= x[i - 2] and (x[i] + x[i - 4] >= x[i - 2]) and (x[i - 1] <= x[i - 3]) and (x[i - 5] + x[i - 1] >= x[i - 3]):
                return True
        return False