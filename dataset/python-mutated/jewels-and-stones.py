class Solution(object):

    def numJewelsInStones(self, J, S):
        if False:
            while True:
                i = 10
        '\n        :type J: str\n        :type S: str\n        :rtype: int\n        '
        lookup = set(J)
        return sum((s in lookup for s in S))