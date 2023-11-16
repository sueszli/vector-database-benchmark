class Solution(object):

    def canCross(self, stones):
        if False:
            return 10
        '\n        :type stones: List[int]\n        :rtype: bool\n        '
        if stones[1] != 1:
            return False
        last_jump_units = {s: set() for s in stones}
        last_jump_units[1].add(1)
        for s in stones[:-1]:
            for j in last_jump_units[s]:
                for k in (j - 1, j, j + 1):
                    if k > 0 and s + k in last_jump_units:
                        last_jump_units[s + k].add(k)
        return bool(last_jump_units[stones[-1]])