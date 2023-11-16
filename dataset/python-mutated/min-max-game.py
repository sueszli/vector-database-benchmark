class Solution(object):

    def minMaxGame(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        n = len(nums)
        while n != 1:
            new_q = []
            for i in xrange(n // 2):
                nums[i] = min(nums[2 * i], nums[2 * i + 1]) if i % 2 == 0 else max(nums[2 * i], nums[2 * i + 1])
            n //= 2
        return nums[0]

class Solution2(object):

    def minMaxGame(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        q = nums[:]
        while len(q) != 1:
            new_q = []
            for i in xrange(len(q) // 2):
                new_q.append(min(q[2 * i], q[2 * i + 1]) if i % 2 == 0 else max(q[2 * i], q[2 * i + 1]))
            q = new_q
        return q[0]