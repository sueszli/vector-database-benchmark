class Solution(object):

    def allPathsSourceTarget(self, graph):
        if False:
            return 10
        '\n        :type graph: List[List[int]]\n        :rtype: List[List[int]]\n        '

        def dfs(graph, curr, path, result):
            if False:
                return 10
            if curr == len(graph) - 1:
                result.append(path[:])
                return
            for node in graph[curr]:
                path.append(node)
                dfs(graph, node, path, result)
                path.pop()
        result = []
        dfs(graph, 0, [0], result)
        return result