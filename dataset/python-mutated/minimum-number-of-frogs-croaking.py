class Solution(object):

    def minNumberOfFrogs(self, croakOfFrogs):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type croakOfFrogs: str\n        :rtype: int\n        '
        S = 'croak'
        lookup = [0] * len(S)
        result = 0
        for c in croakOfFrogs:
            i = S.find(c)
            lookup[i] += 1
            if lookup[i - 1]:
                lookup[i - 1] -= 1
            elif i == 0:
                result += 1
            else:
                return -1
        return result if result == lookup[-1] else -1