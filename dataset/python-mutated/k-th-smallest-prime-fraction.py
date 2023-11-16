class Solution(object):

    def kthSmallestPrimeFraction(self, A, K):
        if False:
            i = 10
            return i + 15
        '\n        :type A: List[int]\n        :type K: int\n        :rtype: List[int]\n        '

        def check(mid, A, K, result):
            if False:
                for i in range(10):
                    print('nop')
            tmp = [0] * 2
            count = 0
            j = 0
            for i in xrange(len(A)):
                while j < len(A):
                    if i < j and A[i] < A[j] * mid:
                        if tmp[0] == 0 or tmp[0] * A[j] < tmp[1] * A[i]:
                            tmp[0] = A[i]
                            tmp[1] = A[j]
                        break
                    j += 1
                count += len(A) - j
            if count == K:
                result[:] = tmp
            return count >= K
        result = []
        (left, right) = (0.0, 1.0)
        while right - left > 1e-08:
            mid = left + (right - left) / 2.0
            if check(mid, A, K, result):
                right = mid
            else:
                left = mid
            if result:
                break
        return result