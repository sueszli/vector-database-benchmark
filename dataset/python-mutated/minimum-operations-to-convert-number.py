class Solution(object):

    def minimumOperations(self, nums, start, goal):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type start: int\n        :type goal: int\n        :rtype: int\n        '
        MAX_X = 1000
        nums = [y for y in nums if y and any((0 <= nx <= MAX_X for nx in (y, goal - y, goal + y, goal ^ y)))]
        q = [(start, 0)]
        lookup = {start}
        while q:
            new_q = []
            for (x, steps) in q:
                for y in nums:
                    for nx in (x + y, x - y, x ^ y):
                        if nx == goal:
                            return steps + 1
                        if not 0 <= nx <= MAX_X or nx in lookup:
                            continue
                        lookup.add(nx)
                        q.append((nx, steps + 1))
            q = new_q
        return -1