class Solution(object):

    def getModifiedArray(self, length, updates):
        if False:
            while True:
                i = 10
        '\n        :type length: int\n        :type updates: List[List[int]]\n        :rtype: List[int]\n        '
        result = [0] * length
        for update in updates:
            result[update[0]] += update[2]
            if update[1] + 1 < length:
                result[update[1] + 1] -= update[2]
        for i in xrange(1, length):
            result[i] += result[i - 1]
        return result