"""
Given a n*n adjacency array.
it will give you a maximum flow.
This version use BFS to search path.

Assume the first is the source and the last is the sink.

Time complexity - O(Ef)

example

graph = [[0, 16, 13, 0, 0, 0],
        [0, 0, 10, 12, 0, 0],
        [0, 4, 0, 0, 14, 0],
        [0, 0, 9, 0, 0, 20],
        [0, 0, 0, 7, 0, 4],
        [0, 0, 0, 0, 0, 0]]

answer should be

23

"""
import copy
import queue
import math

def maximum_flow_bfs(adjacency_matrix):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the maximum flow through a graph using a breadth first search\n    '
    new_array = copy.deepcopy(adjacency_matrix)
    total = 0
    while True:
        min_flow = math.inf
        visited = [0] * len(new_array)
        path = [0] * len(new_array)
        bfs = queue.Queue()
        visited[0] = 1
        bfs.put(0)
        while bfs.qsize() > 0:
            src = bfs.get()
            for k in range(len(new_array)):
                if new_array[src][k] > 0 and visited[k] == 0:
                    visited[k] = 1
                    bfs.put(k)
                    path[k] = src
        if visited[len(new_array) - 1] == 0:
            break
        tmp = len(new_array) - 1
        while tmp != 0:
            if min_flow > new_array[path[tmp]][tmp]:
                min_flow = new_array[path[tmp]][tmp]
            tmp = path[tmp]
        tmp = len(new_array) - 1
        while tmp != 0:
            new_array[path[tmp]][tmp] = new_array[path[tmp]][tmp] - min_flow
            tmp = path[tmp]
        total = total + min_flow
    return total