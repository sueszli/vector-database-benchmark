import collections

class Solution(object):

    def minTime(self, n, edges, hasApple):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type hasApple: List[bool]\n        :rtype: int\n        '
        graph = collections.defaultdict(list)
        for (u, v) in edges:
            graph[u].append(v)
            graph[v].append(u)
        result = [0, 0]
        s = [(1, (-1, 0, result))]
        while s:
            (step, params) = s.pop()
            if step == 1:
                (par, node, ret) = params
                ret[:] = [0, int(hasApple[node])]
                for nei in reversed(graph[node]):
                    if nei == par:
                        continue
                    new_ret = [0, 0]
                    s.append((2, (new_ret, ret)))
                    s.append((1, (node, nei, new_ret)))
            else:
                (new_ret, ret) = params
                ret[0] += new_ret[0] + new_ret[1]
                ret[1] |= bool(new_ret[0] + new_ret[1])
        return 2 * result[0]

class Solution_Recu(object):

    def minTime(self, n, edges, hasApple):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type hasApple: List[bool]\n        :rtype: int\n        '

        def dfs(graph, par, node, hasApple):
            if False:
                while True:
                    i = 10
            (result, extra) = (0, int(hasApple[node]))
            for nei in graph[node]:
                if nei == par:
                    continue
                (count, found) = dfs(graph, node, nei, hasApple)
                result += count + found
                extra |= bool(count + found)
            return (result, extra)
        graph = collections.defaultdict(list)
        for (u, v) in edges:
            graph[u].append(v)
            graph[v].append(u)
        return 2 * dfs(graph, -1, 0, hasApple)[0]

class Solution2(object):

    def minTime(self, n, edges, hasApple):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type hasApple: List[bool]\n        :rtype: int\n        '
        graph = collections.defaultdict(list)
        for (u, v) in edges:
            graph[u].append(v)
            graph[v].append(u)
        result = [0]
        s = [(1, (-1, 0, result))]
        while s:
            (step, params) = s.pop()
            if step == 1:
                (par, node, ret) = params
                tmp = [int(hasApple[node])]
                s.append((3, (tmp, ret)))
                for nei in reversed(graph[node]):
                    if nei == par:
                        continue
                    new_ret = [0]
                    s.append((2, (new_ret, tmp, ret)))
                    s.append((1, (node, nei, new_ret)))
            elif step == 2:
                (new_ret, tmp, ret) = params
                ret[0] += new_ret[0]
                tmp[0] |= bool(new_ret[0])
            else:
                (tmp, ret) = params
                ret[0] += tmp[0]
        return 2 * max(result[0] - 1, 0)

class Solution2_Recu(object):

    def minTime(self, n, edges, hasApple):
        if False:
            return 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type hasApple: List[bool]\n        :rtype: int\n        '

        def dfs(graph, par, node, has_subtree):
            if False:
                return 10
            (result, extra) = (0, int(hasApple[node]))
            for nei in graph[node]:
                if nei == par:
                    continue
                count = dfs(graph, node, nei, hasApple)
                result += count
                extra |= bool(count)
            return result + extra
        graph = collections.defaultdict(list)
        for (u, v) in edges:
            graph[u].append(v)
            graph[v].append(u)
        return 2 * max(dfs(graph, -1, 0, hasApple) - 1, 0)