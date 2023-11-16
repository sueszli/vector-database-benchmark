class Solution(object):

    def hIndex(self, citations):
        if False:
            return 10
        '\n        :type citations: List[int]\n        :rtype: int\n        '
        n = len(citations)
        (left, right) = (0, n - 1)
        while left <= right:
            mid = (left + right) / 2
            if citations[mid] >= n - mid:
                right = mid - 1
            else:
                left = mid + 1
        return n - left