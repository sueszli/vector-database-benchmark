class Solution(object):

    def searchMatrix(self, matrix, target):
        if False:
            print('Hello World!')
        '\n        :type matrix: List[List[int]]\n        :type target: int\n        :rtype: bool\n        '
        if not matrix:
            return False
        (m, n) = (len(matrix), len(matrix[0]))
        (left, right) = (0, m * n)
        while left < right:
            mid = left + (right - left) / 2
            if matrix[mid / n][mid % n] >= target:
                right = mid
            else:
                left = mid + 1
        return left < m * n and matrix[left / n][left % n] == target