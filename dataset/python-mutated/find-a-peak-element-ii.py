class Solution(object):

    def findPeakGrid(self, mat):
        if False:
            return 10
        '\n        :type mat: List[List[int]]\n        :rtype: List[int]\n        '

        def get_vec(mat, i):
            if False:
                i = 10
                return i + 15
            return mat[i] if len(mat) > len(mat[0]) else (mat[j][i] for j in xrange(len(mat)))

        def check(mat, x):
            if False:
                return 10
            return max(get_vec(mat, x)) > max(get_vec(mat, x + 1))
        (left, right) = (0, max(len(mat), len(mat[0])) - 1 - 1)
        while left <= right:
            mid = left + (right - left) // 2
            if check(mat, mid):
                right = mid - 1
            else:
                left = mid + 1
        mav_val = max(get_vec(mat, left))
        result = [left, next((i for (i, x) in enumerate(get_vec(mat, left)) if x == mav_val))]
        return result if len(mat) > len(mat[0]) else result[::-1]