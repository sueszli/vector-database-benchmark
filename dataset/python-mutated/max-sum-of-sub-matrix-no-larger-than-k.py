from bisect import bisect_left, insort

class Solution(object):

    def maxSumSubmatrix(self, matrix, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type matrix: List[List[int]]\n        :type k: int\n        :rtype: int\n        '
        if not matrix:
            return 0
        m = min(len(matrix), len(matrix[0]))
        n = max(len(matrix), len(matrix[0]))
        result = float('-inf')
        for i in xrange(m):
            sums = [0] * n
            for j in xrange(i, m):
                for l in xrange(n):
                    sums[l] += matrix[j][l] if m == len(matrix) else matrix[l][j]
                (accu_sum_set, accu_sum) = ([0], 0)
                for sum in sums:
                    accu_sum += sum
                    it = bisect_left(accu_sum_set, accu_sum - k)
                    if it != len(accu_sum_set):
                        result = max(result, accu_sum - accu_sum_set[it])
                    insort(accu_sum_set, accu_sum)
        return result

class Solution_TLE(object):

    def maxSumSubmatrix(self, matrix, k):
        if False:
            while True:
                i = 10
        '\n        :type matrix: List[List[int]]\n        :type k: int\n        :rtype: int\n        '

        class BST(object):

            def __init__(self, val):
                if False:
                    for i in range(10):
                        print('nop')
                self.val = val
                self.left = None
                self.right = None

            def insert(self, val):
                if False:
                    for i in range(10):
                        print('nop')
                curr = self
                while curr:
                    if curr.val >= val:
                        if curr.left:
                            curr = curr.left
                        else:
                            curr.left = BST(val)
                            return
                    elif curr.right:
                        curr = curr.right
                    else:
                        curr.right = BST(val)
                        return

            def lower_bound(self, val):
                if False:
                    while True:
                        i = 10
                (result, curr) = (None, self)
                while curr:
                    if curr.val >= val:
                        (result, curr) = (curr, curr.left)
                    else:
                        curr = curr.right
                return result
        if not matrix:
            return 0
        m = min(len(matrix), len(matrix[0]))
        n = max(len(matrix), len(matrix[0]))
        result = float('-inf')
        for i in xrange(m):
            sums = [0] * n
            for j in xrange(i, m):
                for l in xrange(n):
                    sums[l] += matrix[j][l] if m == len(matrix) else matrix[l][j]
                accu_sum_set = BST(0)
                accu_sum = 0
                for sum in sums:
                    accu_sum += sum
                    node = accu_sum_set.lower_bound(accu_sum - k)
                    if node:
                        result = max(result, accu_sum - node.val)
                    accu_sum_set.insert(accu_sum)
        return result