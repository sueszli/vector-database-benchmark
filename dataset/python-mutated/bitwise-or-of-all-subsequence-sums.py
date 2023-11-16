class Solution(object):

    def subsequenceSumOr(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = prefix = 0
        for x in nums:
            prefix += x
            result |= x | prefix
        return result

class Solution2(object):

    def subsequenceSumOr(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        result = cnt = 0
        for i in xrange(64):
            cnt >>= 1
            for x in nums:
                cnt += x >> i & 1
            if cnt:
                result |= 1 << i
        return result