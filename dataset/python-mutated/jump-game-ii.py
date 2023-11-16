class Solution(object):

    def jump(self, A):
        if False:
            for i in range(10):
                print('nop')
        jump_count = 0
        reachable = 0
        curr_reachable = 0
        for (i, length) in enumerate(A):
            if i > reachable:
                return -1
            if i > curr_reachable:
                curr_reachable = reachable
                jump_count += 1
            reachable = max(reachable, i + length)
        return jump_count