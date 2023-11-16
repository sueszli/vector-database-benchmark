import collections

class Solution(object):

    def findMatrix(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: List[List[int]]\n        '
        result = []
        cnt = collections.Counter()
        for x in nums:
            if cnt[x] == len(result):
                result.append([])
            result[cnt[x]].append(x)
            cnt[x] += 1
        return result
import collections

class Solution2(object):

    def findMatrix(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: List[List[int]]\n        '
        result = []
        cnt = collections.Counter(nums)
        while cnt:
            result.append(cnt.keys())
            cnt = {k: v - 1 for (k, v) in cnt.iteritems() if v - 1}
        return result