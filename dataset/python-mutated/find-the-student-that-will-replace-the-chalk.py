class Solution(object):

    def chalkReplacer(self, chalk, k):
        if False:
            while True:
                i = 10
        '\n        :type chalk: List[int]\n        :type k: int\n        :rtype: int\n        '
        k %= sum(chalk)
        for (i, x) in enumerate(chalk):
            if k < x:
                return i
            k -= x
        return -1