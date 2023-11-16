import collections

class Solution(object):

    def kthDistinct(self, arr, k):
        if False:
            print('Hello World!')
        '\n        :type arr: List[str]\n        :type k: int\n        :rtype: str\n        '
        count = collections.Counter(arr)
        arr = [x for x in arr if count[x] == 1]
        return arr[k - 1] if k - 1 < len(arr) else ''