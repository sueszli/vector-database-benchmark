class Solution(object):

    def findLonely(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        cnt = collections.Counter(nums)
        return [x for x in nums if cnt[x] == 1 and x - 1 not in cnt and (x + 1 not in cnt)]