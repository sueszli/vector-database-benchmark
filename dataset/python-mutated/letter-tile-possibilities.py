import collections

class Solution(object):

    def numTilePossibilities(self, tiles):
        if False:
            print('Hello World!')
        '\n        :type tiles: str\n        :rtype: int\n        '
        fact = [0.0] * (len(tiles) + 1)
        fact[0] = 1.0
        for i in xrange(1, len(tiles) + 1):
            fact[i] = fact[i - 1] * i
        count = collections.Counter(tiles)
        coeff = [0.0] * (len(tiles) + 1)
        coeff[0] = 1.0
        for i in count.itervalues():
            new_coeff = [0.0] * (len(tiles) + 1)
            for j in xrange(len(coeff)):
                for k in xrange(i + 1):
                    if k + j >= len(new_coeff):
                        break
                    new_coeff[j + k] += coeff[j] * 1.0 / fact[k]
            coeff = new_coeff
        result = 0
        for i in xrange(1, len(coeff)):
            result += int(round(coeff[i] * fact[i]))
        return result

class Solution2(object):

    def numTilePossibilities(self, tiles):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type tiles: str\n        :rtype: int\n        '

        def backtracking(counter):
            if False:
                print('Hello World!')
            total = 0
            for (k, v) in counter.iteritems():
                if not v:
                    continue
                counter[k] -= 1
                total += 1 + backtracking(counter)
                counter[k] += 1
            return total
        return backtracking(collections.Counter(tiles))