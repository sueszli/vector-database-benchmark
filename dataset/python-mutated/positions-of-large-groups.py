class Solution(object):

    def largeGroupPositions(self, S):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type S: str\n        :rtype: List[List[int]]\n        '
        result = []
        i = 0
        for j in xrange(len(S)):
            if j == len(S) - 1 or S[j] != S[j + 1]:
                if j - i + 1 >= 3:
                    result.append([i, j])
                i = j + 1
        return result