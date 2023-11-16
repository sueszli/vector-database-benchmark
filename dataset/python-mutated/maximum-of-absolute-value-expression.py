class Solution(object):

    def maxAbsValExpr(self, arr1, arr2):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arr1: List[int]\n        :type arr2: List[int]\n        :rtype: int\n        '
        result = 0
        for c1 in [1, -1]:
            for c2 in [1, -1]:
                min_prev = float('inf')
                for i in xrange(len(arr1)):
                    curr = c1 * arr1[i] + c2 * arr2[i] + i
                    result = max(result, curr - min_prev)
                    min_prev = min(min_prev, curr)
        return result

class Solution2(object):

    def maxAbsValExpr(self, arr1, arr2):
        if False:
            print('Hello World!')
        '\n        :type arr1: List[int]\n        :type arr2: List[int]\n        :rtype: int\n        '
        return max((max((c1 * arr1[i] + c2 * arr2[i] + i for i in xrange(len(arr1)))) - min((c1 * arr1[i] + c2 * arr2[i] + i for i in xrange(len(arr1)))) for c1 in [1, -1] for c2 in [1, -1]))