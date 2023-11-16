import collections

class Solution(object):

    def hasPath(self, maze, start, destination):
        if False:
            print('Hello World!')
        '\n        :type maze: List[List[int]]\n        :type start: List[int]\n        :type destination: List[int]\n        :rtype: bool\n        '

        def neighbors(maze, node):
            if False:
                print('Hello World!')
            for (i, j) in [(-1, 0), (0, 1), (0, -1), (1, 0)]:
                (x, y) = node
                while 0 <= x + i < len(maze) and 0 <= y + j < len(maze[0]) and (not maze[x + i][y + j]):
                    x += i
                    y += j
                yield (x, y)
        (start, destination) = (tuple(start), tuple(destination))
        queue = collections.deque([start])
        visited = set()
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            if node == destination:
                return True
            visited.add(node)
            for neighbor in neighbors(maze, node):
                queue.append(neighbor)
        return False