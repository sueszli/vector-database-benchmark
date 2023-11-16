class Solution(object):

    def separateDigits(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        result = []
        for x in reversed(nums):
            while x:
                result.append(x % 10)
                x //= 10
        result.reverse()
        return result

class Solution2(object):

    def separateDigits(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        return [int(c) for x in nums for c in str(x)]