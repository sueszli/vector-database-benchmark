import heapq

class Imperfection:
    """The graph is a Node: [List of nodes]
    Where each item in the list of nodes can also have a node with a list of nodes

    The result is that we can keep track of edges, while also keeping it small

    To calculate current, we push the entire graph to A*

    And it calculates the next node to choose, as well as increasing the size
    of the graph with values

    We're using a heap, meaning the element at [0] is always the smallest element

    So we choose that and return it.


    The current A* implementation has an end, we simply do not let it end as LC will make it
    end far before it reaches Searcher again.

    Current is the start position, so if we say we always start at the start of the graph it'll
    go through the entire graph

    graph = {
            Node: [
                {Node :
                {
                    node
                    }
                }
                ]
            }

    For encodings we just do them straight out

    The last value of parents from abstract
    """
    "\n\n   graph = {'A': ['B', 'C'],\n             'B': ['C', 'D'],\n             'C': ['D'],\n             'D': ['C'],\n             'E': ['F'],\n             'F': ['C']}"

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        None

    def findBestNode(self, nodes):
        if False:
            i = 10
            return i + 15
        'Finds the best decryption module'
        return next(iter(nodes))

    def aStar(self, graph, current, end):
        if False:
            print('Hello World!')
        print(f'The graph is {graph}\nCurrent is {current}\n and End is {end}')
        openSet = set()
        openHeap = []
        closedSet = set()

        def retracePath(c):
            if False:
                while True:
                    i = 10
            print('Calling retrace path')
            path = [c]
            while c.parent is not None:
                c = c.parent
                path.append(c)
            path.reverse()
            return path
        print('\n')
        openSet.add(current)
        openHeap.append((0, current))
        while openSet:
            print(f'Openset is {openSet}')
            print(f'OpenHeap is {openHeap}')
            print(f'ClosedSet is {closedSet}')
            print(f'Current is {current}')
            print(f'I am popping {openHeap} with the first element')
            current = heapq.heappop(openHeap)[1]
            print(f'Current is now {current}')
            print(f'Graph current is {graph[current]}')
            if current == end:
                return retracePath(current)
            openSet.remove(current)
            closedSet.add(current)
            for tile in graph[current]:
                if tile not in closedSet:
                    tile.H = (abs(end.x - tile.x) + abs(end.y - tile.y)) * 10
                    tile.H = 1
                    if tile not in openSet:
                        openSet.add(tile)
                        heapq.heappush(openHeap, (tile.H, tile))
                    tile.parent = current
            print('\n')
        return []

class Node:
    """
    A node has a value associated with it
    Calculated from the heuristic
    """

    def __init__(self, h):
        if False:
            return 10
        self.h = h
        self.x = self.h
        self.y = 0.6

    def __le__(self, node2):
        if False:
            return 10
        return self.x <= node2.x

    def __lt__(self, node2):
        if False:
            i = 10
            return i + 15
        return self.x < node2.x
if __name__ == '__main__':
    obj = Imperfection()
    graph = {'A': ['B', 'C'], 'B': ['C', 'D'], 'C': ['D'], 'D': ['C'], 'E': ['F'], 'F': ['C']}
    y = Node(0.5)
    x = Node(0.3)
    p = Node(0.7)
    q = Node(0.9)
    graph = {y: [x, p], p: q}
    print(obj.aStar(graph, y, q))