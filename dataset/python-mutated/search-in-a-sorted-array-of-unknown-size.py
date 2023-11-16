class Solution(object):

    def search(self, reader, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type reader: ArrayReader\n        :type target: int\n        :rtype: int\n        '
        (left, right) = (0, 19999)
        while left <= right:
            mid = left + (right - left) // 2
            response = reader.get(mid)
            if response > target:
                right = mid - 1
            elif response < target:
                left = mid + 1
            else:
                return mid
        return -1