class Solution(object):

    def replaceElements(self, arr):
        if False:
            while True:
                i = 10
        '\n        :type arr: List[int]\n        :rtype: List[int]\n        '
        curr_max = -1
        for i in reversed(xrange(len(arr))):
            (arr[i], curr_max) = (curr_max, max(curr_max, arr[i]))
        return arr