class Solution(object):

    def canJump(self, A):
        if False:
            return 10
        reachable = 0
        for (i, length) in enumerate(A):
            if i > reachable:
                break
            reachable = max(reachable, i + length)
        return reachable >= len(A) - 1