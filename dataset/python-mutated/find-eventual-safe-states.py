class Solution(object):

    def eventualSafeNodes(self, graph):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type graph: List[List[int]]\n        :rtype: List[int]\n        '
        (WHITE, GRAY, BLACK) = range(3)

        def dfs(graph, node, lookup):
            if False:
                return 10
            if lookup[node] != WHITE:
                return lookup[node] == BLACK
            lookup[node] = GRAY
            if any((not dfs(graph, child, lookup) for child in graph[node])):
                return False
            lookup[node] = BLACK
            return True
        lookup = [WHITE] * len(graph)
        return filter(lambda node: dfs(graph, node, lookup), xrange(len(graph)))