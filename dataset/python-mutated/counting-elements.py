class Solution(object):

    def countElements(self, arr):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arr: List[int]\n        :rtype: int\n        '
        lookup = set(arr)
        return sum((1 for x in arr if x + 1 in lookup))

class Solution(object):

    def countElements(self, arr):
        if False:
            return 10
        '\n        :type arr: List[int]\n        :rtype: int\n        '
        arr.sort()
        (result, l) = (0, 1)
        for i in xrange(len(arr) - 1):
            if arr[i] == arr[i + 1]:
                l += 1
                continue
            if arr[i] + 1 == arr[i + 1]:
                result += l
            l = 1
        return result