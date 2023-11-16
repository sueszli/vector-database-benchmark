class Solution(object):

    def maxChunksToSorted(self, arr):
        if False:
            i = 10
            return i + 15
        '\n        :type arr: List[int]\n        :rtype: int\n        '
        (result, max_i) = (0, 0)
        for (i, v) in enumerate(arr):
            max_i = max(max_i, v)
            if max_i == i:
                result += 1
        return result

class Solution2(object):

    def maxChunksToSorted(self, arr):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type arr: List[int]\n        :rtype: int\n        '
        (result, increasing_stk) = (0, [])
        for num in arr:
            max_num = num if not increasing_stk else max(increasing_stk[-1], num)
            while increasing_stk and increasing_stk[-1] > num:
                increasing_stk.pop()
            increasing_stk.append(max_num)
        return len(increasing_stk)