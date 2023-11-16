import string

class Solution(object):

    def twoEditWords(self, queries, dictionary):
        if False:
            while True:
                i = 10
        '\n        :type queries: List[str]\n        :type dictionary: List[str]\n        :rtype: List[str]\n        '
        MOD = (1 << 64) - 59
        BASE = 113
        POW = [1]

        def add(a, b):
            if False:
                i = 10
                return i + 15
            return (a + b) % MOD

        def mult(a, b):
            if False:
                while True:
                    i = 10
            return a * b % MOD

        def pow(i):
            if False:
                for i in range(10):
                    print('nop')
            while not i < len(POW):
                POW.append(mult(POW[-1], BASE))
            return POW[i]
        lookup = set()
        for w in dictionary:
            h = reduce(lambda h, i: add(h, mult(ord(w[i]) - ord('a'), pow(i))), xrange(len(w)), 0)
            for (i, c) in enumerate(w):
                for x in string.ascii_lowercase:
                    if x == c:
                        continue
                    lookup.add(add(h, mult(ord(x) - ord(c), pow(i))))
        result = []
        for w in queries:
            h = reduce(lambda h, i: add(h, mult(ord(w[i]) - ord('a'), pow(i))), xrange(len(w)), 0)
            for (i, c) in enumerate(w):
                for x in string.ascii_lowercase:
                    if x == c:
                        continue
                    if add(h, mult(ord(x) - ord(c), pow(i))) in lookup:
                        break
                else:
                    continue
                result.append(w)
                break
        return result
import itertools

class Solution2(object):

    def twoEditWords(self, queries, dictionary):
        if False:
            i = 10
            return i + 15
        '\n        :type queries: List[str]\n        :type dictionary: List[str]\n        :rtype: List[str]\n        '
        return [q for q in queries if any((sum((c1 != c2 for (c1, c2) in itertools.izip(q, d))) <= 2 for d in dictionary))]