import collections

class Solution(object):

    def uniqueOccurrences(self, arr):
        if False:
            while True:
                i = 10
        '\n        :type arr: List[int]\n        :rtype: bool\n        '
        count = collections.Counter(arr)
        lookup = set()
        for v in count.itervalues():
            if v in lookup:
                return False
            lookup.add(v)
        return True

class Solution2(object):

    def uniqueOccurrences(self, arr):
        if False:
            return 10
        '\n        :type arr: List[int]\n        :rtype: bool\n        '
        count = collections.Counter(arr)
        return len(count) == len(set(count.itervalues()))