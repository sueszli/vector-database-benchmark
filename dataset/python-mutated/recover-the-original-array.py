import collections

class Solution(object):

    def recoverArray(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '

        def check(k, cnt, result):
            if False:
                while True:
                    i = 10
            for x in nums:
                if cnt[x] == 0:
                    continue
                if cnt[x + 2 * k] == 0:
                    return False
                cnt[x] -= 1
                cnt[x + 2 * k] -= 1
                result.append(x + k)
            return True
        nums.sort()
        cnt = collections.Counter(nums)
        for i in xrange(1, len(nums) // 2 + 1):
            k = nums[i] - nums[0]
            if k == 0 or k % 2:
                continue
            k //= 2
            result = []
            if check(k, collections.Counter(cnt), result):
                return result
        return []