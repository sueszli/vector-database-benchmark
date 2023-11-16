class BIT(object):

    def __init__(self, n):
        if False:
            return 10
        self.__bit = [0] * (n + 1)

    def add(self, i, val):
        if False:
            print('Hello World!')
        i += 1
        while i < len(self.__bit):
            self.__bit[i] += val
            i += i & -i

    def query(self, i):
        if False:
            i = 10
            return i + 15
        i += 1
        ret = 0
        while i > 0:
            ret += self.__bit[i]
            i -= i & -i
        return ret

class Solution(object):

    def goodTriplets(self, nums1, nums2):
        if False:
            print('Hello World!')
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: int\n        '
        lookup = [0] * len(nums1)
        for (i, x) in enumerate(nums1):
            lookup[x] = i
        result = 0
        bit = BIT(len(nums1))
        for (i, x) in enumerate(nums2):
            smaller = bit.query(lookup[x] - 1)
            larger = len(nums1) - (lookup[x] + 1) - (i - smaller)
            result += smaller * larger
            bit.add(lookup[x], 1)
        return result