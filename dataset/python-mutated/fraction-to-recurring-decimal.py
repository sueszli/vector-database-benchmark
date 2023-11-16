class Solution(object):

    def fractionToDecimal(self, numerator, denominator):
        if False:
            while True:
                i = 10
        '\n        :type numerator: int\n        :type denominator: int\n        :rtype: str\n        '
        result = ''
        if numerator > 0 and denominator < 0 or (numerator < 0 and denominator > 0):
            result = '-'
        (dvd, dvs) = (abs(numerator), abs(denominator))
        result += str(dvd / dvs)
        dvd %= dvs
        if dvd > 0:
            result += '.'
        lookup = {}
        while dvd and dvd not in lookup:
            lookup[dvd] = len(result)
            dvd *= 10
            result += str(dvd / dvs)
            dvd %= dvs
        if dvd in lookup:
            result = result[:lookup[dvd]] + '(' + result[lookup[dvd]:] + ')'
        return result