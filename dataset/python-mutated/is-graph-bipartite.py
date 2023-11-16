class Solution(object):

    def isBipartite(self, graph):
        if False:
            while True:
                i = 10
        '\n        :type graph: List[List[int]]\n        :rtype: bool\n        '
        color = {}
        for node in xrange(len(graph)):
            if node in color:
                continue
            stack = [node]
            color[node] = 0
            while stack:
                curr = stack.pop()
                for neighbor in graph[curr]:
                    if neighbor not in color:
                        stack.append(neighbor)
                        color[neighbor] = color[curr] ^ 1
                    elif color[neighbor] == color[curr]:
                        return False
        return True