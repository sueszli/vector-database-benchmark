import collections

class UnionFind(object):

    def __init__(self, n):
        if False:
            while True:
                i = 10
        self.set = range(n)

    def find_set(self, x):
        if False:
            while True:
                i = 10
        if self.set[x] != x:
            self.set[x] = self.find_set(self.set[x])
        return self.set[x]

    def union_set(self, x, y):
        if False:
            i = 10
            return i + 15
        (x_root, y_root) = map(self.find_set, (x, y))
        if x_root == y_root:
            return False
        self.set[max(x_root, y_root)] = min(x_root, y_root)
        return True

class Solution(object):

    def smallestStringWithSwaps(self, s, pairs):
        if False:
            i = 10
            return i + 15
        '\n        :type s: str\n        :type pairs: List[List[int]]\n        :rtype: str\n        '
        union_find = UnionFind(len(s))
        for (x, y) in pairs:
            union_find.union_set(x, y)
        components = collections.defaultdict(list)
        for i in xrange(len(s)):
            components[union_find.find_set(i)].append(s[i])
        for i in components.iterkeys():
            components[i].sort(reverse=True)
        result = []
        for i in xrange(len(s)):
            result.append(components[union_find.find_set(i)].pop())
        return ''.join(result)
import itertools

class Solution2(object):

    def smallestStringWithSwaps(self, s, pairs):
        if False:
            print('Hello World!')
        '\n        :type s: str\n        :type pairs: List[List[int]]\n        :rtype: str\n        '

        def dfs(i, adj, lookup, component):
            if False:
                print('Hello World!')
            lookup.add(i)
            component.append(i)
            for j in adj[i]:
                if j in lookup:
                    continue
                dfs(j, adj, lookup, component)
        adj = collections.defaultdict(list)
        for (i, j) in pairs:
            adj[i].append(j)
            adj[j].append(i)
        lookup = set()
        result = list(s)
        for i in xrange(len(s)):
            if i in lookup:
                continue
            component = []
            dfs(i, adj, lookup, component)
            component.sort()
            chars = sorted((result[k] for k in component))
            for (comp, char) in itertools.izip(component, chars):
                result[comp] = char
        return ''.join(result)