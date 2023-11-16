class UndirectedGraphNode(object):

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.label = x
        self.neighbors = []

class Solution(object):

    def cloneGraph(self, node):
        if False:
            while True:
                i = 10
        if node is None:
            return None
        cloned_node = UndirectedGraphNode(node.label)
        (cloned, queue) = ({node: cloned_node}, [node])
        while queue:
            current = queue.pop()
            for neighbor in current.neighbors:
                if neighbor not in cloned:
                    queue.append(neighbor)
                    cloned_neighbor = UndirectedGraphNode(neighbor.label)
                    cloned[neighbor] = cloned_neighbor
                cloned[current].neighbors.append(cloned[neighbor])
        return cloned[node]