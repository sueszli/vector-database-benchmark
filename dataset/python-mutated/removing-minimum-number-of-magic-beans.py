class Solution(object):

    def minimumRemoval(self, beans):
        if False:
            while True:
                i = 10
        '\n        :type beans: List[int]\n        :rtype: int\n        '
        beans.sort()
        return sum(beans) - max((x * (len(beans) - i) for (i, x) in enumerate(beans)))