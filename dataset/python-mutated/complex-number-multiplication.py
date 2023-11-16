class Solution(object):

    def complexNumberMultiply(self, a, b):
        if False:
            i = 10
            return i + 15
        '\n        :type a: str\n        :type b: str\n        :rtype: str\n        '
        (ra, ia) = map(int, a[:-1].split('+'))
        (rb, ib) = map(int, b[:-1].split('+'))
        return '%d+%di' % (ra * rb - ia * ib, ra * ib + ia * rb)