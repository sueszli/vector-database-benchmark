from heapq import heappush, heappop

class Solution(object):

    def kthSmallest(self, matrix, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type matrix: List[List[int]]\n        :type k: int\n        :rtype: int\n        '
        kth_smallest = 0
        min_heap = []

        def push(i, j):
            if False:
                print('Hello World!')
            if len(matrix) > len(matrix[0]):
                if i < len(matrix[0]) and j < len(matrix):
                    heappush(min_heap, [matrix[j][i], i, j])
            elif i < len(matrix) and j < len(matrix[0]):
                heappush(min_heap, [matrix[i][j], i, j])
        push(0, 0)
        while min_heap and k > 0:
            (kth_smallest, i, j) = heappop(min_heap)
            push(i, j + 1)
            if j == 0:
                push(i + 1, 0)
            k -= 1
        return kth_smallest