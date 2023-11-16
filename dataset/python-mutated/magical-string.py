import itertools

class Solution(object):

    def magicalString(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: int\n        '

        def gen():
            if False:
                return 10
            for c in (1, 2, 2):
                yield c
            for (i, c) in enumerate(gen()):
                if i > 1:
                    for _ in xrange(c):
                        yield (i % 2 + 1)
        return sum((c & 1 for c in itertools.islice(gen(), n)))