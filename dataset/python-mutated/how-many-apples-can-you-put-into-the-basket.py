class Solution(object):

    def maxNumberOfApples(self, arr):
        if False:
            while True:
                i = 10
        '\n        :type arr: List[int]\n        :rtype: int\n        '
        LIMIT = 5000
        arr.sort()
        (result, total) = (0, 0)
        for x in arr:
            if total + x > LIMIT:
                break
            total += x
            result += 1
        return result