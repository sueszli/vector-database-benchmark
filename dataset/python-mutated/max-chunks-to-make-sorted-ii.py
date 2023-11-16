class Solution(object):

    def maxChunksToSorted(self, arr):
        if False:
            return 10
        '\n        :type arr: List[int]\n        :rtype: int\n        '
        (result, increasing_stk) = (0, [])
        for num in arr:
            max_num = num if not increasing_stk else max(increasing_stk[-1], num)
            while increasing_stk and increasing_stk[-1] > num:
                increasing_stk.pop()
            increasing_stk.append(max_num)
        return len(increasing_stk)

class Solution2(object):

    def maxChunksToSorted(self, arr):
        if False:
            while True:
                i = 10
        '\n        :type arr: List[int]\n        :rtype: int\n        '

        def compare(i1, i2):
            if False:
                while True:
                    i = 10
            return arr[i1] - arr[i2] if arr[i1] != arr[i2] else i1 - i2
        idxs = [i for i in xrange(len(arr))]
        (result, max_i) = (0, 0)
        for (i, v) in enumerate(sorted(idxs, cmp=compare)):
            max_i = max(max_i, v)
            if max_i == i:
                result += 1
        return result