class Solution(object):

    def removeDuplicates(self, A):
        if False:
            while True:
                i = 10
        if not A:
            return 0
        (last, i, same) = (0, 1, False)
        while i < len(A):
            if A[last] != A[i] or not same:
                same = A[last] == A[i]
                last += 1
                A[last] = A[i]
            i += 1
        return last + 1