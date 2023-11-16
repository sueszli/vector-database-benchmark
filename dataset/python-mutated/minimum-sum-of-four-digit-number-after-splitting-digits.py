class Solution(object):

    def minimumSum(self, num):
        if False:
            while True:
                i = 10
        '\n        :type num: int\n        :rtype: int\n        '

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
        nums = map(int, list(str(num)))
        inplace_counting_sort(nums)
        a = b = 0
        for x in nums:
            a = a * 10 + x
            (a, b) = (b, a)
        return a + b

class Solution2(object):

    def minimumSum(self, num):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type num: int\n        :rtype: int\n        '
        nums = sorted(map(int, list(str(num))))
        a = b = 0
        for x in nums:
            a = a * 10 + x
            (a, b) = (b, a)
        return a + b