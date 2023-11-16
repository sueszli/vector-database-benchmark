class Solution(object):

    def leastOpsExpressTarget(self, x, target):
        if False:
            print('Hello World!')
        '\n        :type x: int\n        :type target: int\n        :rtype: int\n        '
        (pos, neg, k) = (0, 0, 0)
        while target:
            (target, r) = divmod(target, x)
            if k:
                (pos, neg) = (min(r * k + pos, (r + 1) * k + neg), min((x - r) * k + pos, (x - r - 1) * k + neg))
            else:
                (pos, neg) = (r * 2, (x - r) * 2)
            k += 1
        return min(pos, k + neg) - 1