class Solution(object):

    def findDifferentBinaryString(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[str]\n        :rtype: str\n        '
        return ''.join(('01'[nums[i][i] == '0'] for i in xrange(len(nums))))

class Solution2(object):

    def findDifferentBinaryString(self, nums):
        if False:
            return 10
        '\n        :type nums: List[str]\n        :rtype: str\n        '
        lookup = set(map(lambda x: int(x, 2), nums))
        return next((bin(i)[2:].zfill(len(nums[0])) for i in xrange(2 ** len(nums[0])) if i not in lookup))

class Solution_Extra(object):

    def findAllDifferentBinaryStrings(self, nums):
        if False:
            return 10
        '\n        :type nums: List[str]\n        :rtype: List[str]\n        '
        lookup = set(map(lambda x: int(x, 2), nums))
        return [bin(i)[2:].zfill(len(nums[0])) for i in xrange(2 ** len(nums[0])) if i not in lookup]