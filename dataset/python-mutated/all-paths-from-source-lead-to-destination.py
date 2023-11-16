import collections

class Solution(object):

    def leadsToDestination(self, n, edges, source, destination):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type source: int\n        :type destination: int\n        :rtype: bool\n        '
        (UNVISITED, VISITING, DONE) = range(3)

        def dfs(children, node, destination, status):
            if False:
                while True:
                    i = 10
            if status[node] == DONE:
                return True
            if status[node] == VISITING:
                return False
            status[node] = VISITING
            if node not in children and node != destination:
                return False
            if node in children:
                for child in children[node]:
                    if not dfs(children, child, destination, status):
                        return False
            status[node] = DONE
            return True
        children = collections.defaultdict(list)
        for (parent, child) in edges:
            children[parent].append(child)
        return dfs(children, source, destination, [0] * n)