class Solution(object):

    def isArmstrong(self, N):
        if False:
            i = 10
            return i + 15
        '\n        :type N: int\n        :rtype: bool\n        '
        n_str = str(N)
        return sum((int(i) ** len(n_str) for i in n_str)) == N