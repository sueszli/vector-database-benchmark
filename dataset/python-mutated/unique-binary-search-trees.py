class Solution(object):

    def numTrees(self, n):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :rtype: int\n        '
        if n == 0:
            return 1

        def combination(n, k):
            if False:
                return 10
            count = 1
            for i in xrange(1, k + 1):
                count = count * (n - i + 1) / i
            return count
        return combination(2 * n, n) - combination(2 * n, n - 1)

class Solution2(object):

    def numTrees(self, n):
        if False:
            return 10
        counts = [1, 1]
        for i in xrange(2, n + 1):
            count = 0
            for j in xrange(i):
                count += counts[j] * counts[i - j - 1]
            counts.append(count)
        return counts[-1]