class Solution(object):

    def maxIncreasingGroups(self, usageLimits):
        if False:
            i = 10
            return i + 15
        '\n        :type usageLimits: List[int]\n        :rtype: int\n        '

        def inplace_counting_sort(nums, reverse=False):
            if False:
                print('Hello World!')
            if not nums:
                return
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
        usageLimits = [min(x, len(usageLimits)) for x in usageLimits]
        inplace_counting_sort(usageLimits)
        result = curr = 0
        for x in usageLimits:
            curr += x
            if curr >= result + 1:
                curr -= result + 1
                result += 1
        return result

class Solution2(object):

    def maxIncreasingGroups(self, usageLimits):
        if False:
            i = 10
            return i + 15
        '\n        :type usageLimits: List[int]\n        :rtype: int\n        '
        usageLimits.sort()
        result = curr = 0
        for x in usageLimits:
            curr += x
            if curr >= result + 1:
                curr -= result + 1
                result += 1
        return result

class Solution3(object):

    def maxIncreasingGroups(self, usageLimits):
        if False:
            i = 10
            return i + 15
        '\n        :type usageLimits: List[int]\n        :rtype: int\n        '

        def check(l):
            if False:
                print('Hello World!')
            curr = 0
            for i in xrange(l):
                curr += usageLimits[~i] - (l - i)
                curr = min(curr, 0)
            for i in xrange(len(usageLimits) - l):
                curr += usageLimits[i]
            return curr >= 0
        usageLimits.sort()
        (left, right) = (1, len(usageLimits))
        while left <= right:
            mid = left + (right - left) // 2
            if not check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return right

class Solution4(object):

    def maxIncreasingGroups(self, usageLimits):
        if False:
            while True:
                i = 10
        '\n        :type usageLimits: List[int]\n        :rtype: int\n        '

        def check(l):
            if False:
                i = 10
                return i + 15
            return all(((i + 1) * i // 2 <= prefix[len(usageLimits) - (l - i)] for i in xrange(1, l + 1)))
        usageLimits.sort()
        prefix = [0] * (len(usageLimits) + 1)
        for i in xrange(len(usageLimits)):
            prefix[i + 1] = prefix[i] + usageLimits[i]
        (left, right) = (1, len(usageLimits))
        while left <= right:
            mid = left + (right - left) // 2
            if not check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return right