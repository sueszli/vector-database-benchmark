class Solution(object):

    def findSubsequences(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: List[List[int]]\n        '

        def findSubsequencesHelper(nums, pos, seq, result):
            if False:
                i = 10
                return i + 15
            if len(seq) >= 2:
                result.append(list(seq))
            lookup = set()
            for i in xrange(pos, len(nums)):
                if (not seq or nums[i] >= seq[-1]) and nums[i] not in lookup:
                    lookup.add(nums[i])
                    seq.append(nums[i])
                    findSubsequencesHelper(nums, i + 1, seq, result)
                    seq.pop()
        (result, seq) = ([], [])
        findSubsequencesHelper(nums, 0, seq, result)
        return result