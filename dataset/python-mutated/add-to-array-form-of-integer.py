class Solution(object):

    def addToArrayForm(self, A, K):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type A: List[int]\n        :type K: int\n        :rtype: List[int]\n        '
        A.reverse()
        (carry, i) = (K, 0)
        A[i] += carry
        (carry, A[i]) = divmod(A[i], 10)
        while carry:
            i += 1
            if i < len(A):
                A[i] += carry
            else:
                A.append(carry)
            (carry, A[i]) = divmod(A[i], 10)
        A.reverse()
        return A