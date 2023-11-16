class Solution(object):

    def xorQueries(self, arr, queries):
        if False:
            while True:
                i = 10
        '\n        :type arr: List[int]\n        :type queries: List[List[int]]\n        :rtype: List[int]\n        '
        for i in xrange(1, len(arr)):
            arr[i] ^= arr[i - 1]
        return [arr[right] ^ arr[left - 1] if left else arr[right] for (left, right) in queries]