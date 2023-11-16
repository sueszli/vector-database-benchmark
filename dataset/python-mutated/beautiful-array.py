class Solution(object):

    def beautifulArray(self, N):
        if False:
            return 10
        '\n        :type N: int\n        :rtype: List[int]\n        '
        result = [1]
        while len(result) < N:
            result = [i * 2 - 1 for i in result] + [i * 2 for i in result]
        return [i for i in result if i <= N]