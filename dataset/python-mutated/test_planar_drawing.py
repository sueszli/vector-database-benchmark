import math
import pytest
import networkx as nx
from networkx.algorithms.planar_drawing import triangulate_embedding

def test_graph1():
    if False:
        i = 10
        return i + 15
    embedding_data = {0: [1, 2, 3], 1: [2, 0], 2: [3, 0, 1], 3: [2, 0]}
    check_embedding_data(embedding_data)

def test_graph2():
    if False:
        for i in range(10):
            print('nop')
    embedding_data = {0: [8, 6], 1: [2, 6, 9], 2: [8, 1, 7, 9, 6, 4], 3: [9], 4: [2], 5: [6, 8], 6: [9, 1, 0, 5, 2], 7: [9, 2], 8: [0, 2, 5], 9: [1, 6, 2, 7, 3]}
    check_embedding_data(embedding_data)

def test_circle_graph():
    if False:
        i = 10
        return i + 15
    embedding_data = {0: [1, 9], 1: [0, 2], 2: [1, 3], 3: [2, 4], 4: [3, 5], 5: [4, 6], 6: [5, 7], 7: [6, 8], 8: [7, 9], 9: [8, 0]}
    check_embedding_data(embedding_data)

def test_grid_graph():
    if False:
        return 10
    embedding_data = {(0, 1): [(0, 0), (1, 1), (0, 2)], (1, 2): [(1, 1), (2, 2), (0, 2)], (0, 0): [(0, 1), (1, 0)], (2, 1): [(2, 0), (2, 2), (1, 1)], (1, 1): [(2, 1), (1, 2), (0, 1), (1, 0)], (2, 0): [(1, 0), (2, 1)], (2, 2): [(1, 2), (2, 1)], (1, 0): [(0, 0), (2, 0), (1, 1)], (0, 2): [(1, 2), (0, 1)]}
    check_embedding_data(embedding_data)

def test_one_node_graph():
    if False:
        while True:
            i = 10
    embedding_data = {0: []}
    check_embedding_data(embedding_data)

def test_two_node_graph():
    if False:
        i = 10
        return i + 15
    embedding_data = {0: [1], 1: [0]}
    check_embedding_data(embedding_data)

def test_three_node_graph():
    if False:
        for i in range(10):
            print('nop')
    embedding_data = {0: [1, 2], 1: [0, 2], 2: [0, 1]}
    check_embedding_data(embedding_data)

def test_multiple_component_graph1():
    if False:
        while True:
            i = 10
    embedding_data = {0: [], 1: []}
    check_embedding_data(embedding_data)

def test_multiple_component_graph2():
    if False:
        while True:
            i = 10
    embedding_data = {0: [1, 2], 1: [0, 2], 2: [0, 1], 3: [4, 5], 4: [3, 5], 5: [3, 4]}
    check_embedding_data(embedding_data)

def test_invalid_half_edge():
    if False:
        i = 10
        return i + 15
    with pytest.raises(nx.NetworkXException):
        embedding_data = {1: [2, 3, 4], 2: [1, 3, 4], 3: [1, 2, 4], 4: [1, 2, 3]}
        embedding = nx.PlanarEmbedding()
        embedding.set_data(embedding_data)
        nx.combinatorial_embedding_to_pos(embedding)

def test_triangulate_embedding1():
    if False:
        while True:
            i = 10
    embedding = nx.PlanarEmbedding()
    embedding.add_node(1)
    expected_embedding = {1: []}
    check_triangulation(embedding, expected_embedding)

def test_triangulate_embedding2():
    if False:
        return 10
    embedding = nx.PlanarEmbedding()
    embedding.connect_components(1, 2)
    expected_embedding = {1: [2], 2: [1]}
    check_triangulation(embedding, expected_embedding)

def check_triangulation(embedding, expected_embedding):
    if False:
        while True:
            i = 10
    (res_embedding, _) = triangulate_embedding(embedding, True)
    assert res_embedding.get_data() == expected_embedding, 'Expected embedding incorrect'
    (res_embedding, _) = triangulate_embedding(embedding, False)
    assert res_embedding.get_data() == expected_embedding, 'Expected embedding incorrect'

def check_embedding_data(embedding_data):
    if False:
        print('Hello World!')
    'Checks that the planar embedding of the input is correct'
    embedding = nx.PlanarEmbedding()
    embedding.set_data(embedding_data)
    pos_fully = nx.combinatorial_embedding_to_pos(embedding, False)
    msg = 'Planar drawing does not conform to the embedding (fully triangulation)'
    assert planar_drawing_conforms_to_embedding(embedding, pos_fully), msg
    check_edge_intersections(embedding, pos_fully)
    pos_internally = nx.combinatorial_embedding_to_pos(embedding, True)
    msg = 'Planar drawing does not conform to the embedding (internal triangulation)'
    assert planar_drawing_conforms_to_embedding(embedding, pos_internally), msg
    check_edge_intersections(embedding, pos_internally)

def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    if False:
        print('Hello World!')
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def point_in_between(a, b, p):
    if False:
        return 10
    (x1, y1) = a
    (x2, y2) = b
    (px, py) = p
    dist_1_2 = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    dist_1_p = math.sqrt((x1 - px) ** 2 + (y1 - py) ** 2)
    dist_2_p = math.sqrt((x2 - px) ** 2 + (y2 - py) ** 2)
    return is_close(dist_1_p + dist_2_p, dist_1_2)

def check_edge_intersections(G, pos):
    if False:
        for i in range(10):
            print('nop')
    'Check all edges in G for intersections.\n\n    Raises an exception if an intersection is found.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n    pos : dict\n        Maps every node to a tuple (x, y) representing its position\n\n    '
    for (a, b) in G.edges():
        for (c, d) in G.edges():
            if a != c and b != d and (b != c) and (a != d):
                (x1, y1) = pos[a]
                (x2, y2) = pos[b]
                (x3, y3) = pos[c]
                (x4, y4) = pos[d]
                determinant = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if determinant != 0:
                    px = (x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4) / determinant
                    py = (x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4) / determinant
                    if point_in_between(pos[a], pos[b], (px, py)) and point_in_between(pos[c], pos[d], (px, py)):
                        msg = f'There is an intersection at {px},{py}'
                        raise nx.NetworkXException(msg)
                msg = 'A node lies on a edge connecting two other nodes'
                if point_in_between(pos[a], pos[b], pos[c]) or point_in_between(pos[a], pos[b], pos[d]) or point_in_between(pos[c], pos[d], pos[a]) or point_in_between(pos[c], pos[d], pos[b]):
                    raise nx.NetworkXException(msg)

class Vector:
    """Compare vectors by their angle without loss of precision

    All vectors in direction [0, 1] are the smallest.
    The vectors grow in clockwise direction.
    """
    __slots__ = ['x', 'y', 'node', 'quadrant']

    def __init__(self, x, y, node):
        if False:
            i = 10
            return i + 15
        self.x = x
        self.y = y
        self.node = node
        if self.x >= 0 and self.y > 0:
            self.quadrant = 1
        elif self.x > 0 and self.y <= 0:
            self.quadrant = 2
        elif self.x <= 0 and self.y < 0:
            self.quadrant = 3
        else:
            self.quadrant = 4

    def __eq__(self, other):
        if False:
            while True:
                i = 10
        return self.quadrant == other.quadrant and self.x * other.y == self.y * other.x

    def __lt__(self, other):
        if False:
            while True:
                i = 10
        if self.quadrant < other.quadrant:
            return True
        elif self.quadrant > other.quadrant:
            return False
        else:
            return self.x * other.y < self.y * other.x

    def __ne__(self, other):
        if False:
            print('Hello World!')
        return self != other

    def __le__(self, other):
        if False:
            print('Hello World!')
        return not other < self

    def __gt__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return other < self

    def __ge__(self, other):
        if False:
            print('Hello World!')
        return not self < other

def planar_drawing_conforms_to_embedding(embedding, pos):
    if False:
        while True:
            i = 10
    'Checks if pos conforms to the planar embedding\n\n    Returns true iff the neighbors are actually oriented in the orientation\n    specified of the embedding\n    '
    for v in embedding:
        nbr_vectors = []
        v_pos = pos[v]
        for nbr in embedding[v]:
            new_vector = Vector(pos[nbr][0] - v_pos[0], pos[nbr][1] - v_pos[1], nbr)
            nbr_vectors.append(new_vector)
        nbr_vectors.sort()
        for (idx, nbr_vector) in enumerate(nbr_vectors):
            cw_vector = nbr_vectors[(idx + 1) % len(nbr_vectors)]
            ccw_vector = nbr_vectors[idx - 1]
            if embedding[v][nbr_vector.node]['cw'] != cw_vector.node or embedding[v][nbr_vector.node]['ccw'] != ccw_vector.node:
                return False
            if cw_vector.node != nbr_vector.node and cw_vector == nbr_vector:
                return False
            if ccw_vector.node != nbr_vector.node and ccw_vector == nbr_vector:
                return False
    return True