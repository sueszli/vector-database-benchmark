class Solution(object):

    def loudAndRich(self, richer, quiet):
        if False:
            print('Hello World!')
        '\n        :type richer: List[List[int]]\n        :type quiet: List[int]\n        :rtype: List[int]\n        '

        def dfs(graph, quiet, node, result):
            if False:
                for i in range(10):
                    print('nop')
            if result[node] is None:
                result[node] = node
                for nei in graph[node]:
                    smallest_person = dfs(graph, quiet, nei, result)
                    if quiet[smallest_person] < quiet[result[node]]:
                        result[node] = smallest_person
            return result[node]
        graph = [[] for _ in xrange(len(quiet))]
        for (u, v) in richer:
            graph[v].append(u)
        result = [None] * len(quiet)
        return map(lambda x: dfs(graph, quiet, x, result), xrange(len(quiet)))