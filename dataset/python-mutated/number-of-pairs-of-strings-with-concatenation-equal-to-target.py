import collections

class Solution(object):

    def numOfPairs(self, nums, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[str]\n        :type target: str\n        :rtype: int\n        '
        lookup = collections.Counter()
        result = 0
        for num in nums:
            (cnt1, cnt2) = (lookup[-(len(target) - len(num))], lookup[len(target) - len(num)])
            if target.startswith(num):
                result += cnt1
                lookup[len(num)] += 1
            if target.endswith(num):
                result += cnt2
                lookup[-len(num)] += 1
        return result
import collections

class Solution2(object):

    def numOfPairs(self, nums, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[str]\n        :type target: str\n        :rtype: int\n        '
        (prefix, suffix) = (collections.Counter(), collections.Counter())
        result = 0
        for num in nums:
            if target.startswith(num):
                result += suffix[len(target) - len(num)]
            if target.endswith(num):
                result += prefix[len(target) - len(num)]
            if target.startswith(num):
                prefix[len(num)] += 1
            if target.endswith(num):
                suffix[len(num)] += 1
        return result