class Solution(object):

    def largestAltitude(self, gain):
        if False:
            print('Hello World!')
        '\n        :type gain: List[int]\n        :rtype: int\n        '
        result = curr = 0
        for g in gain:
            curr += g
            result = max(result, curr)
        return result