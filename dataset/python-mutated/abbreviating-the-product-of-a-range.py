import math

class Solution(object):

    def abbreviateProduct(self, left, right):
        if False:
            return 10
        '\n        :type left: int\n        :type right: int\n        :rtype: str\n        '
        PREFIX_LEN = SUFFIX_LEN = 5
        MOD = 10 ** (PREFIX_LEN + SUFFIX_LEN)
        (curr, zeros) = (1, 0)
        abbr = False
        for i in xrange(left, right + 1):
            curr *= i
            while not curr % 10:
                curr //= 10
                zeros += 1
            (q, curr) = divmod(curr, MOD)
            if q:
                abbr = True
        if not abbr:
            return '%se%s' % (curr, zeros)
        decimal = reduce(lambda x, y: (x + y) % 1, (math.log10(i) for i in xrange(left, right + 1)))
        prefix = str(int(10 ** (decimal + (PREFIX_LEN - 1))))
        suffix = str(curr % 10 ** SUFFIX_LEN).zfill(SUFFIX_LEN)
        return '%s...%se%s' % (prefix, suffix, zeros)