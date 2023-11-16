class ArrayReader(object):

    def compareSub(self, l, r, x, y):
        if False:
            for i in range(10):
                print('nop')
        pass

    def length(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class Solution(object):

    def getIndex(self, reader):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type reader: ArrayReader\n        :rtype: integer\n        '
        (left, right) = (0, reader.length() - 1)
        while left < right:
            mid = left + (right - left) // 2
            if reader.compareSub(left, mid, mid if (right - left + 1) % 2 else mid + 1, right) >= 0:
                right = mid
            else:
                left = mid + 1
        return left