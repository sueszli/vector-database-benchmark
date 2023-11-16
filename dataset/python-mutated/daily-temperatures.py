class Solution(object):

    def dailyTemperatures(self, temperatures):
        if False:
            return 10
        '\n        :type temperatures: List[int]\n        :rtype: List[int]\n        '
        result = [0] * len(temperatures)
        stk = []
        for i in xrange(len(temperatures)):
            while stk and temperatures[stk[-1]] < temperatures[i]:
                idx = stk.pop()
                result[idx] = i - idx
            stk.append(i)
        return result