class Solution(object):

    def canMeasureWater(self, x, y, z):
        if False:
            print('Hello World!')
        '\n        :type x: int\n        :type y: int\n        :type z: int\n        :rtype: bool\n        '

        def gcd(a, b):
            if False:
                for i in range(10):
                    print('nop')
            while b:
                (a, b) = (b, a % b)
            return a
        return z == 0 or (z <= x + y and z % gcd(x, y) == 0)