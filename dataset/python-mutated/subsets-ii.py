class Solution(object):

    def subsetsWithDup(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: List[List[int]]\n        '
        nums.sort()
        result = [[]]
        previous_size = 0
        for i in xrange(len(nums)):
            size = len(result)
            for j in xrange(size):
                if i == 0 or nums[i] != nums[i - 1] or j >= previous_size:
                    result.append(list(result[j]))
                    result[-1].append(nums[i])
            previous_size = size
        return result

class Solution2(object):

    def subsetsWithDup(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: List[List[int]]\n        '
        result = []
        (i, count) = (0, 1 << len(nums))
        nums.sort()
        while i < count:
            cur = []
            for j in xrange(len(nums)):
                if i & 1 << j:
                    cur.append(nums[j])
            if cur not in result:
                result.append(cur)
            i += 1
        return result

class Solution3(object):

    def subsetsWithDup(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: List[List[int]]\n        '
        result = []
        self.subsetsWithDupRecu(result, [], sorted(nums))
        return result

    def subsetsWithDupRecu(self, result, cur, nums):
        if False:
            while True:
                i = 10
        if not nums:
            if cur not in result:
                result.append(cur)
        else:
            self.subsetsWithDupRecu(result, cur, nums[1:])
            self.subsetsWithDupRecu(result, cur + [nums[0]], nums[1:])