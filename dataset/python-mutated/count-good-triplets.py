class Solution(object):

    def countGoodTriplets(self, arr, a, b, c):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arr: List[int]\n        :type a: int\n        :type b: int\n        :type c: int\n        :rtype: int\n        '
        return sum((abs(arr[i] - arr[j]) <= a and abs(arr[j] - arr[k]) <= b and (abs(arr[k] - arr[i]) <= c) for i in xrange(len(arr) - 2) for j in xrange(i + 1, len(arr) - 1) for k in xrange(j + 1, len(arr))))