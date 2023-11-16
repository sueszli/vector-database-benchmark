class Solution(object):

    def getLastMoment(self, n, left, right):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type left: List[int]\n        :type right: List[int]\n        :rtype: int\n        '
        return max(max(left or [0]), n - min(right or [n]))