class Solution(object):

    def arrayRankTransform(self, arr):
        if False:
            return 10
        '\n        :type arr: List[int]\n        :rtype: List[int]\n        '
        return map({x: i + 1 for (i, x) in enumerate(sorted(set(arr)))}.get, arr)