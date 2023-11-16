"""
Given the capacity, source and sink of a graph,
computes the maximum flow from source to sink.
Input : capacity, source, sink
Output : maximum flow from source to sink
Capacity is a two-dimensional array that is v*v.
capacity[i][j] implies the capacity of the edge from i to j.
If there is no edge from i to j, capacity[i][j] should be zero.
"""
from queue import Queue

def dfs(capacity, flow, visit, vertices, idx, sink, current_flow=1 << 63):
    if False:
        return 10
    '\n    Depth First Search implementation for Ford-Fulkerson algorithm.\n    '
    if idx == sink:
        return current_flow
    visit[idx] = True
    for nxt in range(vertices):
        if not visit[nxt] and flow[idx][nxt] < capacity[idx][nxt]:
            available_flow = min(current_flow, capacity[idx][nxt] - flow[idx][nxt])
            tmp = dfs(capacity, flow, visit, vertices, nxt, sink, available_flow)
            if tmp:
                flow[idx][nxt] += tmp
                flow[nxt][idx] -= tmp
                return tmp
    return 0

def ford_fulkerson(capacity, source, sink):
    if False:
        return 10
    '\n    Computes maximum flow from source to sink using DFS.\n    Time Complexity : O(Ef)\n    E is the number of edges and f is the maximum flow in the graph.\n    '
    vertices = len(capacity)
    ret = 0
    flow = [[0] * vertices for _ in range(vertices)]
    while True:
        visit = [False for _ in range(vertices)]
        tmp = dfs(capacity, flow, visit, vertices, source, sink)
        if tmp:
            ret += tmp
        else:
            break
    return ret

def edmonds_karp(capacity, source, sink):
    if False:
        for i in range(10):
            print('nop')
    '\n    Computes maximum flow from source to sink using BFS.\n    Time complexity : O(V*E^2)\n    V is the number of vertices and E is the number of edges.\n    '
    vertices = len(capacity)
    ret = 0
    flow = [[0] * vertices for _ in range(vertices)]
    while True:
        tmp = 0
        queue = Queue()
        visit = [False for _ in range(vertices)]
        par = [-1 for _ in range(vertices)]
        visit[source] = True
        queue.put((source, 1 << 63))
        while queue.qsize():
            front = queue.get()
            (idx, current_flow) = front
            if idx == sink:
                tmp = current_flow
                break
            for nxt in range(vertices):
                if not visit[nxt] and flow[idx][nxt] < capacity[idx][nxt]:
                    visit[nxt] = True
                    par[nxt] = idx
                    queue.put((nxt, min(current_flow, capacity[idx][nxt] - flow[idx][nxt])))
        if par[sink] == -1:
            break
        ret += tmp
        parent = par[sink]
        idx = sink
        while parent != -1:
            flow[parent][idx] += tmp
            flow[idx][parent] -= tmp
            idx = parent
            parent = par[parent]
    return ret

def dinic_bfs(capacity, flow, level, source, sink):
    if False:
        return 10
    '\n    BFS function for Dinic algorithm.\n    Check whether sink is reachable only using edges that is not full.\n    '
    vertices = len(capacity)
    queue = Queue()
    queue.put(source)
    level[source] = 0
    while queue.qsize():
        front = queue.get()
        for nxt in range(vertices):
            if level[nxt] == -1 and flow[front][nxt] < capacity[front][nxt]:
                level[nxt] = level[front] + 1
                queue.put(nxt)
    return level[sink] != -1

def dinic_dfs(capacity, flow, level, idx, sink, work, current_flow=1 << 63):
    if False:
        i = 10
        return i + 15
    '\n    DFS function for Dinic algorithm.\n    Finds new flow using edges that is not full.\n    '
    if idx == sink:
        return current_flow
    vertices = len(capacity)
    while work[idx] < vertices:
        nxt = work[idx]
        if level[nxt] == level[idx] + 1 and flow[idx][nxt] < capacity[idx][nxt]:
            available_flow = min(current_flow, capacity[idx][nxt] - flow[idx][nxt])
            tmp = dinic_dfs(capacity, flow, level, nxt, sink, work, available_flow)
            if tmp > 0:
                flow[idx][nxt] += tmp
                flow[nxt][idx] -= tmp
                return tmp
        work[idx] += 1
    return 0

def dinic(capacity, source, sink):
    if False:
        return 10
    '\n    Computes maximum flow from source to sink using Dinic algorithm.\n    Time complexity : O(V^2*E)\n    V is the number of vertices and E is the number of edges.\n    '
    vertices = len(capacity)
    flow = [[0] * vertices for i in range(vertices)]
    ret = 0
    while True:
        level = [-1 for i in range(vertices)]
        work = [0 for i in range(vertices)]
        if not dinic_bfs(capacity, flow, level, source, sink):
            break
        while True:
            tmp = dinic_dfs(capacity, flow, level, source, sink, work)
            if tmp > 0:
                ret += tmp
            else:
                break
    return ret