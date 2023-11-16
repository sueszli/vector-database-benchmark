class Solution(object):

    def removeElement(self, A, elem):
        if False:
            while True:
                i = 10
        (i, last) = (0, len(A) - 1)
        while i <= last:
            if A[i] == elem:
                (A[i], A[last]) = (A[last], A[i])
                last -= 1
            else:
                i += 1
        return last + 1