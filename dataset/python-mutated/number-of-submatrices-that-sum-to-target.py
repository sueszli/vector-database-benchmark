import collections

class Solution(object):

    def numSubmatrixSumTarget(self, matrix, target):
        if False:
            while True:
                i = 10
        '\n        :type matrix: List[List[int]]\n        :type target: int\n        :rtype: int\n        '
        if len(matrix) > len(matrix[0]):
            return self.numSubmatrixSumTarget(map(list, zip(*matrix)), target)
        for i in xrange(len(matrix)):
            for j in xrange(len(matrix[i]) - 1):
                matrix[i][j + 1] += matrix[i][j]
        result = 0
        for i in xrange(len(matrix)):
            prefix_sum = [0] * len(matrix[i])
            for j in xrange(i, len(matrix)):
                lookup = collections.defaultdict(int)
                lookup[0] = 1
                for k in xrange(len(matrix[j])):
                    prefix_sum[k] += matrix[j][k]
                    if prefix_sum[k] - target in lookup:
                        result += lookup[prefix_sum[k] - target]
                    lookup[prefix_sum[k]] += 1
        return result