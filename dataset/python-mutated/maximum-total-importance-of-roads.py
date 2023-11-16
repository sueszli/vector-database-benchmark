class Solution(object):

    def maximumImportance(self, n, roads):
        if False:
            return 10
        '\n        :type n: int\n        :type roads: List[List[int]]\n        :rtype: int\n        '

        def inplace_counting_sort(nums, reverse=False):
            if False:
                i = 10
                return i + 15
            count = [0] * (max(nums) + 1)
            for num in nums:
                count[num] += 1
            for i in xrange(1, len(count)):
                count[i] += count[i - 1]
            for i in reversed(xrange(len(nums))):
                while nums[i] >= 0:
                    count[nums[i]] -= 1
                    j = count[nums[i]]
                    (nums[i], nums[j]) = (nums[j], ~nums[i])
            for i in xrange(len(nums)):
                nums[i] = ~nums[i]
            if reverse:
                nums.reverse()
        degree = [0] * n
        for (a, b) in roads:
            degree[a] += 1
            degree[b] += 1
        inplace_counting_sort(degree)
        return sum((i * x for (i, x) in enumerate(degree, 1)))

class Solution2(object):

    def maximumImportance(self, n, roads):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type roads: List[List[int]]\n        :rtype: int\n        '
        degree = [0] * n
        for (a, b) in roads:
            degree[a] += 1
            degree[b] += 1
        degree.sort()
        return sum((i * x for (i, x) in enumerate(degree, 1)))