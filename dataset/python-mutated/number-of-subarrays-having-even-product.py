class Solution(object):

    def evenProduct(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = (len(nums) + 1) * len(nums) // 2
        cnt = 0
        for x in nums:
            cnt = cnt + 1 if x % 2 else 0
            result -= cnt
        return result

class Solution2(object):

    def evenProduct(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = cnt = 0
        for (i, x) in enumerate(nums):
            if x % 2 == 0:
                cnt = i + 1
            result += cnt
        return result