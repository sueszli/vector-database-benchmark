import collections

class Solution(object):

    def isPossibleDivide(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: bool\n        '
        count = collections.Counter(nums)
        for num in sorted(count.keys()):
            c = count[num]
            if not c:
                continue
            for i in xrange(num, num + k):
                if count[i] < c:
                    return False
                count[i] -= c
        return True