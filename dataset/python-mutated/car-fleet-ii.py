class Solution(object):

    def getCollisionTimes(self, cars):
        if False:
            print('Hello World!')
        '\n        :type cars: List[List[int]]\n        :rtype: List[float]\n        '
        stk = []
        result = [-1.0] * len(cars)
        for i in reversed(xrange(len(cars))):
            (p, s) = cars[i]
            while stk and (cars[stk[-1]][1] >= s or 0 < result[stk[-1]] <= float(cars[stk[-1]][0] - p) / (s - cars[stk[-1]][1])):
                stk.pop()
            if stk:
                result[i] = float(cars[stk[-1]][0] - p) / (s - cars[stk[-1]][1])
            stk.append(i)
        return result