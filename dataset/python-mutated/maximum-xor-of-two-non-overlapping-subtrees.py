class Trie(object):

    def __init__(self, bit_length):
        if False:
            print('Hello World!')
        self.__root = {}
        self.__bit_length = bit_length

    def insert(self, num):
        if False:
            return 10
        node = self.__root
        for i in reversed(xrange(self.__bit_length)):
            curr = num >> i & 1
            if curr not in node:
                node[curr] = {}
            node = node[curr]

    def query(self, num):
        if False:
            for i in range(10):
                print('nop')
        if not self.__root:
            return -1
        (node, result) = (self.__root, 0)
        for i in reversed(xrange(self.__bit_length)):
            curr = num >> i & 1
            if 1 ^ curr in node:
                node = node[1 ^ curr]
                result |= 1 << i
            else:
                node = node[curr]
        return result

class Solution(object):

    def maxXor(self, n, edges, values):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type values: List[int]\n        :rtype: int\n        '

        def iter_dfs():
            if False:
                print('Hello World!')
            lookup = [0] * len(values)
            stk = [(1, 0, -1)]
            while stk:
                (step, u, p) = stk.pop()
                if step == 1:
                    stk.append((2, u, p))
                    for v in adj[u]:
                        if v == p:
                            continue
                        stk.append((1, v, u))
                elif step == 2:
                    lookup[u] = values[u] + sum((lookup[v] for v in adj[u] if v != p))
            return lookup

        def iter_dfs2():
            if False:
                while True:
                    i = 10
            trie = Trie(lookup[0].bit_length())
            result = [0]
            stk = [(1, (0, -1, result))]
            while stk:
                (step, args) = stk.pop()
                if step == 1:
                    (u, p, ret) = args
                    ret[0] = max(trie.query(lookup[u]), 0)
                    stk.append((3, (u,)))
                    for v in adj[u]:
                        if v == p:
                            continue
                        new_ret = [0]
                        stk.append((2, (new_ret, ret)))
                        stk.append((1, (v, u, new_ret)))
                elif step == 2:
                    (new_ret, ret) = args
                    ret[0] = max(ret[0], new_ret[0])
                elif step == 3:
                    u = args[0]
                    trie.insert(lookup[u])
            return result[0]
        adj = [[] for _ in xrange(len(values))]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        lookup = iter_dfs()
        return iter_dfs2()

class Trie(object):

    def __init__(self, bit_length):
        if False:
            print('Hello World!')
        self.__root = {}
        self.__bit_length = bit_length

    def insert(self, num):
        if False:
            return 10
        node = self.__root
        for i in reversed(xrange(self.__bit_length)):
            curr = num >> i & 1
            if curr not in node:
                node[curr] = {}
            node = node[curr]

    def query(self, num):
        if False:
            return 10
        if not self.__root:
            return -1
        (node, result) = (self.__root, 0)
        for i in reversed(xrange(self.__bit_length)):
            curr = num >> i & 1
            if 1 ^ curr in node:
                node = node[1 ^ curr]
                result |= 1 << i
            else:
                node = node[curr]
        return result

class Solution2(object):

    def maxXor(self, n, edges, values):
        if False:
            return 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type values: List[int]\n        :rtype: int\n        '

        def dfs(u, p):
            if False:
                return 10
            lookup[u] = values[u] + sum((dfs(v, u) for v in adj[u] if v != p))
            return lookup[u]

        def dfs2(u, p):
            if False:
                return 10
            result = max(trie.query(lookup[u]), 0)
            for v in adj[u]:
                if v == p:
                    continue
                result = max(result, dfs2(v, u))
            trie.insert(lookup[u])
            return result
        adj = [[] for _ in xrange(len(values))]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        lookup = [0] * len(values)
        dfs(0, -1)
        trie = Trie(lookup[0].bit_length())
        return dfs2(0, -1)