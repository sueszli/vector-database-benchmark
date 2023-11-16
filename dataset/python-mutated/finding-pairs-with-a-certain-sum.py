import collections

class FindSumPairs(object):

    def __init__(self, nums1, nums2):
        if False:
            i = 10
            return i + 15
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        '
        self.__nums2 = nums2
        self.__count1 = collections.Counter(nums1)
        self.__count2 = collections.Counter(nums2)

    def add(self, index, val):
        if False:
            while True:
                i = 10
        '\n        :type index: int\n        :type val: int\n        :rtype: None\n        '
        self.__count2[self.__nums2[index]] -= 1
        self.__nums2[index] += val
        self.__count2[self.__nums2[index]] += 1

    def count(self, tot):
        if False:
            i = 10
            return i + 15
        '\n        :type tot: int\n        :rtype: int\n        '
        return sum((cnt * self.__count2[tot - x] for (x, cnt) in self.__count1.iteritems()))