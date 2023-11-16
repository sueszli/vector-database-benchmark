class Solution(object):

    def hammingDistance(self, x, y):
        if False:
            while True:
                i = 10
        '\n        :type x: int\n        :type y: int\n        :rtype: int\n        '
        distance = 0
        z = x ^ y
        while z:
            distance += 1
            z &= z - 1
        return distance

    def hammingDistance2(self, x, y):
        if False:
            while True:
                i = 10
        '\n        :type x: int\n        :type y: int\n        :rtype: int\n        '
        return bin(x ^ y).count('1')