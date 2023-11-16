import fractions

class Solution(object):

    def countDifferentSubsequenceGCDs(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (max_num, nums_set) = (max(nums), set(nums))
        result = 0
        for i in xrange(1, max_num + 1):
            d = 0
            for x in xrange(i, max_num + 1, i):
                if x not in nums_set:
                    continue
                d = fractions.gcd(d, x)
                if d == i:
                    result += 1
                    break
        return result