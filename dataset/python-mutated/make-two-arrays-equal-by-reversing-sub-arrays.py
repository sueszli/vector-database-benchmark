import collections

class Solution(object):

    def canBeEqual(self, target, arr):
        if False:
            return 10
        '\n        :type target: List[int]\n        :type arr: List[int]\n        :rtype: bool\n        '
        return collections.Counter(target) == collections.Counter(arr)

class Solution2(object):

    def canBeEqual(self, target, arr):
        if False:
            while True:
                i = 10
        '\n        :type target: List[int]\n        :type arr: List[int]\n        :rtype: bool\n        '
        (target.sort(), arr.sort())
        return target == arr