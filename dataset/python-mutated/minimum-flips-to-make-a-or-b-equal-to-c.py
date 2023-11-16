class Solution(object):

    def minFlips(self, a, b, c):
        if False:
            while True:
                i = 10
        '\n        :type a: int\n        :type b: int\n        :type c: int\n        :rtype: int\n        '

        def number_of_1_bits(n):
            if False:
                while True:
                    i = 10
            result = 0
            while n:
                n &= n - 1
                result += 1
            return result
        return number_of_1_bits((a | b) ^ c) + number_of_1_bits(a & b & ~c)

class Solution2(object):

    def minFlips(self, a, b, c):
        if False:
            while True:
                i = 10
        '\n        :type a: int\n        :type b: int\n        :type c: int\n        :rtype: int\n        '
        result = 0
        for i in xrange(31):
            (a_i, b_i, c_i) = map(lambda x: x & 1, [a, b, c])
            if a_i | b_i != c_i:
                result += 2 if a_i == b_i == 1 else 1
            (a, b, c) = (a >> 1, b >> 1, c >> 1)
        return result