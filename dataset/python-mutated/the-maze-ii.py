import heapq

class Solution(object):

    def shortestDistance(self, maze, start, destination):
        if False:
            i = 10
            return i + 15
        '\n        :type maze: List[List[int]]\n        :type start: List[int]\n        :type destination: List[int]\n        :rtype: int\n        '
        (start, destination) = (tuple(start), tuple(destination))

        def neighbors(maze, node):
            if False:
                return 10
            for dir in [(-1, 0), (0, 1), (0, -1), (1, 0)]:
                (cur_node, dist) = (list(node), 0)
                while 0 <= cur_node[0] + dir[0] < len(maze) and 0 <= cur_node[1] + dir[1] < len(maze[0]) and (not maze[cur_node[0] + dir[0]][cur_node[1] + dir[1]]):
                    cur_node[0] += dir[0]
                    cur_node[1] += dir[1]
                    dist += 1
                yield (dist, tuple(cur_node))
        heap = [(0, start)]
        visited = set()
        while heap:
            (dist, node) = heapq.heappop(heap)
            if node in visited:
                continue
            if node == destination:
                return dist
            visited.add(node)
            for (neighbor_dist, neighbor) in neighbors(maze, node):
                heapq.heappush(heap, (dist + neighbor_dist, neighbor))
        return -1