# Time:  O(n + m)
# Space: O(1)

class Solution(object):
    def repeatedStringMatch(self, A, B):
        """
        :type A: str
        :type B: str
        :rtype: int
        """
        def check(index):
            return all(A[(i+index) % len(A)] == c
                       for i, c in enumerate(B))

        M, p = 10**9+7, 113
        p_inv = pow(p, M-2, M)
        q = (len(B)+len(A)-1) // len(A)

        b_hash, power = 0, 1
        for c in B:
            b_hash += power * ord(c)
            b_hash %= M
            power = (power*p) % M

        a_hash, power = 0, 1
        for i in xrange(len(B)):
            a_hash += power * ord(A[i%len(A)])
            a_hash %= M
            power = (power*p) % M

        if a_hash == b_hash and check(0): return q

        power = (power*p_inv) % M
        for i in xrange(len(B), (q+1)*len(A)):
            a_hash = (a_hash-ord(A[(i-len(B))%len(A)])) * p_inv
            a_hash += power * ord(A[i%len(A)])
            a_hash %= M
            if a_hash == b_hash and check(i-len(B)+1):
                return q if i < q*len(A) else q+1

        return -1

