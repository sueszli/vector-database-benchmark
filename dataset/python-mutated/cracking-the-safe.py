class Solution(object):

    def crackSafe(self, n, k):
        if False:
            return 10
        '\n        :type n: int\n        :type k: int\n        :rtype: str\n        '
        M = k ** (n - 1)
        P = [q * k + i for i in xrange(k) for q in xrange(M)]
        result = [str(k - 1)] * (n - 1)
        for i in xrange(k ** n):
            j = i
            while P[j] >= 0:
                result.append(str(j // M))
                (P[j], j) = (-1, P[j])
        return ''.join(result)

class Solution2(object):

    def crackSafe(self, n, k):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type k: int\n        :rtype: str\n        '
        total = k ** n
        M = total // k
        unique_rolling_hash = 0
        result = [str(0)] * (n - 1)
        lookup = set()
        while len(lookup) < total:
            for i in reversed(xrange(k)):
                new_unique_rolling_hash = unique_rolling_hash * k + i
                if new_unique_rolling_hash not in lookup:
                    lookup.add(new_unique_rolling_hash)
                    result.append(str(i))
                    unique_rolling_hash = new_unique_rolling_hash % M
                    break
        return ''.join(result)

class Solution3(object):

    def crackSafe(self, n, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type k: int\n        :rtype: str\n        '
        M = k ** (n - 1)

        def dfs(k, unique_rolling_hash, lookup, result):
            if False:
                return 10
            for i in reversed(xrange(k)):
                new_unique_rolling_hash = unique_rolling_hash * k + i
                if new_unique_rolling_hash not in lookup:
                    lookup.add(new_unique_rolling_hash)
                    result.append(str(i))
                    dfs(k, new_unique_rolling_hash % M, lookup, result)
                    break
        unique_rolling_hash = 0
        result = [str(0)] * (n - 1)
        lookup = set()
        dfs(k, unique_rolling_hash, lookup, result)
        return ''.join(result)

class Solution4(object):

    def crackSafe(self, n, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type k: int\n        :rtype: str\n        '
        result = [str(k - 1)] * (n - 1)
        lookup = set()
        total = k ** n
        while len(lookup) < total:
            node = result[len(result) - n + 1:]
            for i in xrange(k):
                neighbor = ''.join(node) + str(i)
                if neighbor not in lookup:
                    lookup.add(neighbor)
                    result.append(str(i))
                    break
        return ''.join(result)

class Solution5(object):

    def crackSafe(self, n, k):
        if False:
            return 10
        '\n        :type n: int\n        :type k: int\n        :rtype: str\n        '

        def dfs(k, node, lookup, result):
            if False:
                for i in range(10):
                    print('nop')
            for i in xrange(k):
                neighbor = node + str(i)
                if neighbor not in lookup:
                    lookup.add(neighbor)
                    result.append(str(i))
                    dfs(k, neighbor[1:], lookup, result)
                    break
        result = [str(k - 1)] * (n - 1)
        lookup = set()
        dfs(k, ''.join(result), lookup, result)
        return ''.join(result)