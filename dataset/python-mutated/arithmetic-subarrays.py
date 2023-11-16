import itertools

class Solution(object):

    def checkArithmeticSubarrays(self, nums, l, r):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type l: List[int]\n        :type r: List[int]\n        :rtype: List[bool]\n        '

        def is_arith(n):
            if False:
                return 10
            (mx, mn, lookup) = (max(n), min(n), set(n))
            if mx == mn:
                return True
            (d, r) = divmod(mx - mn, len(n) - 1)
            if r:
                return False
            return all((i in lookup for i in xrange(mn, mx, d)))
        result = []
        for (left, right) in itertools.izip(l, r):
            result.append(is_arith(nums[left:right + 1]))
        return result