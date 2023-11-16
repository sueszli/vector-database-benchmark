class Solution(object):

    def relativeSortArray(self, arr1, arr2):
        if False:
            while True:
                i = 10
        '\n        :type arr1: List[int]\n        :type arr2: List[int]\n        :rtype: List[int]\n        '
        lookup = {v: i for (i, v) in enumerate(arr2)}
        return sorted(arr1, key=lambda i: lookup.get(i, len(arr2) + i))