class Solution(object):

    def transformArray(self, arr):
        if False:
            i = 10
            return i + 15
        '\n        :type arr: List[int]\n        :rtype: List[int]\n        '

        def is_changable(arr):
            if False:
                i = 10
                return i + 15
            return any((arr[i - 1] > arr[i] < arr[i + 1] or arr[i - 1] < arr[i] > arr[i + 1] for i in xrange(1, len(arr) - 1)))
        while is_changable(arr):
            new_arr = arr[:]
            for i in xrange(1, len(arr) - 1):
                new_arr[i] += arr[i - 1] > arr[i] < arr[i + 1]
                new_arr[i] -= arr[i - 1] < arr[i] > arr[i + 1]
            arr = new_arr
        return arr