import itertools

class Solution(object):

    def relocateMarbles(self, nums, moveFrom, moveTo):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type moveFrom: List[int]\n        :type moveTo: List[int]\n        :rtype: List[int]\n        '
        lookup = set(nums)
        for (a, b) in itertools.izip(moveFrom, moveTo):
            lookup.remove(a)
            lookup.add(b)
        return sorted(lookup)