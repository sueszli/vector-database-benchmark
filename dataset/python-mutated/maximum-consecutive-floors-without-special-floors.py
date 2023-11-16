class Solution(object):

    def maxConsecutive(self, bottom, top, special):
        if False:
            print('Hello World!')
        '\n        :type bottom: int\n        :type top: int\n        :type special: List[int]\n        :rtype: int\n        '
        special.sort()
        result = max(special[0] - bottom, top - special[-1])
        for i in xrange(1, len(special)):
            result = max(result, special[i] - special[i - 1] - 1)
        return result