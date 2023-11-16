class Solution(object):

    def isOneBitCharacter(self, bits):
        if False:
            i = 10
            return i + 15
        '\n        :type bits: List[int]\n        :rtype: bool\n        '
        parity = 0
        for i in reversed(xrange(len(bits) - 1)):
            if bits[i] == 0:
                break
            parity ^= bits[i]
        return parity == 0