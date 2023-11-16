class Solution(object):

    def isCovered(self, ranges, left, right):
        if False:
            print('Hello World!')
        '\n        :type ranges: List[List[int]]\n        :type left: int\n        :type right: int\n        :rtype: bool\n        '
        RANGE_SIZE = 50
        interval = [0] * (RANGE_SIZE + 1)
        for (l, r) in ranges:
            interval[l - 1] += 1
            interval[r - 1 + 1] -= 1
        cnt = 0
        for i in xrange(right - 1 + 1):
            cnt += interval[i]
            if i >= left - 1 and (not cnt):
                return False
        return True

class Solution2(object):

    def isCovered(self, ranges, left, right):
        if False:
            return 10
        '\n        :type ranges: List[List[int]]\n        :type left: int\n        :type right: int\n        :rtype: bool\n        '
        ranges.sort()
        for (l, r) in ranges:
            if l <= left <= r:
                left = r + 1
        return left > right

class Solution3(object):

    def isCovered(self, ranges, left, right):
        if False:
            print('Hello World!')
        '\n        :type ranges: List[List[int]]\n        :type left: int\n        :type right: int\n        :rtype: bool\n        '
        return all((any((l <= i <= r for (l, r) in ranges)) for i in xrange(left, right + 1)))