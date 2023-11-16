class Solution(object):

    def mctFromLeafValues(self, arr):
        if False:
            while True:
                i = 10
        '\n        :type arr: List[int]\n        :rtype: int\n        '
        result = 0
        stk = [float('inf')]
        for x in arr:
            while stk[-1] <= x:
                result += stk.pop() * min(stk[-1], x)
            stk.append(x)
        while len(stk) > 2:
            result += stk.pop() * stk[-1]
        return result