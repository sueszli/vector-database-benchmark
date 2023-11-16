class Solution(object):

    def decode(self, encoded, first):
        if False:
            i = 10
            return i + 15
        '\n        :type encoded: List[int]\n        :type first: int\n        :rtype: List[int]\n        '
        result = [first]
        for x in encoded:
            result.append(result[-1] ^ x)
        return result