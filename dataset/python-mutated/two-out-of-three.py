import collections

class Solution(object):

    def twoOutOfThree(self, nums1, nums2, nums3):
        if False:
            print('Hello World!')
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :type nums3: List[int]\n        :rtype: List[int]\n        '
        K = 2
        cnt = collections.Counter()
        for nums in (nums1, nums2, nums3):
            cnt.update(set(nums))
        return [x for (x, c) in cnt.iteritems() if c >= K]
import collections

class Solution2(object):

    def twoOutOfThree(self, nums1, nums2, nums3):
        if False:
            return 10
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :type nums3: List[int]\n        :rtype: List[int]\n        '
        K = 2
        cnt = collections.Counter()
        result = []
        for nums in (nums1, nums2, nums3):
            for x in set(nums):
                cnt[x] += 1
                if cnt[x] == K:
                    result.append(x)
        return result