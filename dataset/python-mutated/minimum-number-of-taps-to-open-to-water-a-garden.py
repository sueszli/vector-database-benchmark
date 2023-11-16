class Solution(object):

    def minTaps(self, n, ranges):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type ranges: List[int]\n        :rtype: int\n        '

        def jump_game(A):
            if False:
                while True:
                    i = 10
            (jump_count, reachable, curr_reachable) = (0, 0, 0)
            for (i, length) in enumerate(A):
                if i > reachable:
                    return -1
                if i > curr_reachable:
                    curr_reachable = reachable
                    jump_count += 1
                reachable = max(reachable, i + length)
            return jump_count
        max_range = [0] * (n + 1)
        for (i, r) in enumerate(ranges):
            (left, right) = (max(i - r, 0), min(i + r, n))
            max_range[left] = max(max_range[left], right - left)
        return jump_game(max_range)