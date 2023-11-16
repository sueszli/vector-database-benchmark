class Solution(object):

    def numberOfArrays(self, differences, lower, upper):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type differences: List[int]\n        :type lower: int\n        :type upper: int\n        :rtype: int\n        '
        total = mn = mx = 0
        for x in differences:
            total += x
            mn = min(mn, total)
            mx = max(mx, total)
        return max(upper - lower - (mx - mn) + 1, 0)