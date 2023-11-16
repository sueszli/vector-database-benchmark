from collections import defaultdict
import networkx as nx
__all__ = ['combinatorial_embedding_to_pos']

def combinatorial_embedding_to_pos(embedding, fully_triangulate=False):
    if False:
        i = 10
        return i + 15
    'Assigns every node a (x, y) position based on the given embedding\n\n    The algorithm iteratively inserts nodes of the input graph in a certain\n    order and rearranges previously inserted nodes so that the planar drawing\n    stays valid. This is done efficiently by only maintaining relative\n    positions during the node placements and calculating the absolute positions\n    at the end. For more information see [1]_.\n\n    Parameters\n    ----------\n    embedding : nx.PlanarEmbedding\n        This defines the order of the edges\n\n    fully_triangulate : bool\n        If set to True the algorithm adds edges to a copy of the input\n        embedding and makes it chordal.\n\n    Returns\n    -------\n    pos : dict\n        Maps each node to a tuple that defines the (x, y) position\n\n    References\n    ----------\n    .. [1] M. Chrobak and T.H. Payne:\n        A Linear-time Algorithm for Drawing a Planar Graph on a Grid 1989\n        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.51.6677\n\n    '
    if len(embedding.nodes()) < 4:
        default_positions = [(0, 0), (2, 0), (1, 1)]
        pos = {}
        for (i, v) in enumerate(embedding.nodes()):
            pos[v] = default_positions[i]
        return pos
    (embedding, outer_face) = triangulate_embedding(embedding, fully_triangulate)
    left_t_child = {}
    right_t_child = {}
    delta_x = {}
    y_coordinate = {}
    node_list = get_canonical_ordering(embedding, outer_face)
    (v1, v2, v3) = (node_list[0][0], node_list[1][0], node_list[2][0])
    delta_x[v1] = 0
    y_coordinate[v1] = 0
    right_t_child[v1] = v3
    left_t_child[v1] = None
    delta_x[v2] = 1
    y_coordinate[v2] = 0
    right_t_child[v2] = None
    left_t_child[v2] = None
    delta_x[v3] = 1
    y_coordinate[v3] = 1
    right_t_child[v3] = v2
    left_t_child[v3] = None
    for k in range(3, len(node_list)):
        (vk, contour_neighbors) = node_list[k]
        wp = contour_neighbors[0]
        wp1 = contour_neighbors[1]
        wq = contour_neighbors[-1]
        wq1 = contour_neighbors[-2]
        adds_mult_tri = len(contour_neighbors) > 2
        delta_x[wp1] += 1
        delta_x[wq] += 1
        delta_x_wp_wq = sum((delta_x[x] for x in contour_neighbors[1:]))
        delta_x[vk] = (-y_coordinate[wp] + delta_x_wp_wq + y_coordinate[wq]) // 2
        y_coordinate[vk] = (y_coordinate[wp] + delta_x_wp_wq + y_coordinate[wq]) // 2
        delta_x[wq] = delta_x_wp_wq - delta_x[vk]
        if adds_mult_tri:
            delta_x[wp1] -= delta_x[vk]
        right_t_child[wp] = vk
        right_t_child[vk] = wq
        if adds_mult_tri:
            left_t_child[vk] = wp1
            right_t_child[wq1] = None
        else:
            left_t_child[vk] = None
    pos = {}
    pos[v1] = (0, y_coordinate[v1])
    remaining_nodes = [v1]
    while remaining_nodes:
        parent_node = remaining_nodes.pop()
        set_position(parent_node, left_t_child, remaining_nodes, delta_x, y_coordinate, pos)
        set_position(parent_node, right_t_child, remaining_nodes, delta_x, y_coordinate, pos)
    return pos

def set_position(parent, tree, remaining_nodes, delta_x, y_coordinate, pos):
    if False:
        while True:
            i = 10
    'Helper method to calculate the absolute position of nodes.'
    child = tree[parent]
    parent_node_x = pos[parent][0]
    if child is not None:
        child_x = parent_node_x + delta_x[child]
        pos[child] = (child_x, y_coordinate[child])
        remaining_nodes.append(child)

def get_canonical_ordering(embedding, outer_face):
    if False:
        while True:
            i = 10
    'Returns a canonical ordering of the nodes\n\n    The canonical ordering of nodes (v1, ..., vn) must fulfill the following\n    conditions:\n    (See Lemma 1 in [2]_)\n\n    - For the subgraph G_k of the input graph induced by v1, ..., vk it holds:\n        - 2-connected\n        - internally triangulated\n        - the edge (v1, v2) is part of the outer face\n    - For a node v(k+1) the following holds:\n        - The node v(k+1) is part of the outer face of G_k\n        - It has at least two neighbors in G_k\n        - All neighbors of v(k+1) in G_k lie consecutively on the outer face of\n          G_k (excluding the edge (v1, v2)).\n\n    The algorithm used here starts with G_n (containing all nodes). It first\n    selects the nodes v1 and v2. And then tries to find the order of the other\n    nodes by checking which node can be removed in order to fulfill the\n    conditions mentioned above. This is done by calculating the number of\n    chords of nodes on the outer face. For more information see [1]_.\n\n    Parameters\n    ----------\n    embedding : nx.PlanarEmbedding\n        The embedding must be triangulated\n    outer_face : list\n        The nodes on the outer face of the graph\n\n    Returns\n    -------\n    ordering : list\n        A list of tuples `(vk, wp_wq)`. Here `vk` is the node at this position\n        in the canonical ordering. The element `wp_wq` is a list of nodes that\n        make up the outer face of G_k.\n\n    References\n    ----------\n    .. [1] Steven Chaplick.\n        Canonical Orders of Planar Graphs and (some of) Their Applications 2015\n        https://wuecampus2.uni-wuerzburg.de/moodle/pluginfile.php/545727/mod_resource/content/0/vg-ss15-vl03-canonical-orders-druckversion.pdf\n    .. [2] M. Chrobak and T.H. Payne:\n        A Linear-time Algorithm for Drawing a Planar Graph on a Grid 1989\n        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.51.6677\n\n    '
    v1 = outer_face[0]
    v2 = outer_face[1]
    chords = defaultdict(int)
    marked_nodes = set()
    ready_to_pick = set(outer_face)
    outer_face_ccw_nbr = {}
    prev_nbr = v2
    for idx in range(2, len(outer_face)):
        outer_face_ccw_nbr[prev_nbr] = outer_face[idx]
        prev_nbr = outer_face[idx]
    outer_face_ccw_nbr[prev_nbr] = v1
    outer_face_cw_nbr = {}
    prev_nbr = v1
    for idx in range(len(outer_face) - 1, 0, -1):
        outer_face_cw_nbr[prev_nbr] = outer_face[idx]
        prev_nbr = outer_face[idx]

    def is_outer_face_nbr(x, y):
        if False:
            while True:
                i = 10
        if x not in outer_face_ccw_nbr:
            return outer_face_cw_nbr[x] == y
        if x not in outer_face_cw_nbr:
            return outer_face_ccw_nbr[x] == y
        return outer_face_ccw_nbr[x] == y or outer_face_cw_nbr[x] == y

    def is_on_outer_face(x):
        if False:
            i = 10
            return i + 15
        return x not in marked_nodes and (x in outer_face_ccw_nbr or x == v1)
    for v in outer_face:
        for nbr in embedding.neighbors_cw_order(v):
            if is_on_outer_face(nbr) and (not is_outer_face_nbr(v, nbr)):
                chords[v] += 1
                ready_to_pick.discard(v)
    canonical_ordering = [None] * len(embedding.nodes())
    canonical_ordering[0] = (v1, [])
    canonical_ordering[1] = (v2, [])
    ready_to_pick.discard(v1)
    ready_to_pick.discard(v2)
    for k in range(len(embedding.nodes()) - 1, 1, -1):
        v = ready_to_pick.pop()
        marked_nodes.add(v)
        wp = None
        wq = None
        nbr_iterator = iter(embedding.neighbors_cw_order(v))
        while True:
            nbr = next(nbr_iterator)
            if nbr in marked_nodes:
                continue
            if is_on_outer_face(nbr):
                if nbr == v1:
                    wp = v1
                elif nbr == v2:
                    wq = v2
                elif outer_face_cw_nbr[nbr] == v:
                    wp = nbr
                else:
                    wq = nbr
            if wp is not None and wq is not None:
                break
        wp_wq = [wp]
        nbr = wp
        while nbr != wq:
            next_nbr = embedding[v][nbr]['ccw']
            wp_wq.append(next_nbr)
            outer_face_cw_nbr[nbr] = next_nbr
            outer_face_ccw_nbr[next_nbr] = nbr
            nbr = next_nbr
        if len(wp_wq) == 2:
            chords[wp] -= 1
            if chords[wp] == 0:
                ready_to_pick.add(wp)
            chords[wq] -= 1
            if chords[wq] == 0:
                ready_to_pick.add(wq)
        else:
            new_face_nodes = set(wp_wq[1:-1])
            for w in new_face_nodes:
                ready_to_pick.add(w)
                for nbr in embedding.neighbors_cw_order(w):
                    if is_on_outer_face(nbr) and (not is_outer_face_nbr(w, nbr)):
                        chords[w] += 1
                        ready_to_pick.discard(w)
                        if nbr not in new_face_nodes:
                            chords[nbr] += 1
                            ready_to_pick.discard(nbr)
        canonical_ordering[k] = (v, wp_wq)
    return canonical_ordering

def triangulate_face(embedding, v1, v2):
    if False:
        while True:
            i = 10
    'Triangulates the face given by half edge (v, w)\n\n    Parameters\n    ----------\n    embedding : nx.PlanarEmbedding\n    v1 : node\n        The half-edge (v1, v2) belongs to the face that gets triangulated\n    v2 : node\n    '
    (_, v3) = embedding.next_face_half_edge(v1, v2)
    (_, v4) = embedding.next_face_half_edge(v2, v3)
    if v1 in (v2, v3):
        return
    while v1 != v4:
        if embedding.has_edge(v1, v3):
            (v1, v2, v3) = (v2, v3, v4)
        else:
            embedding.add_half_edge_cw(v1, v3, v2)
            embedding.add_half_edge_ccw(v3, v1, v2)
            (v1, v2, v3) = (v1, v3, v4)
        (_, v4) = embedding.next_face_half_edge(v2, v3)

def triangulate_embedding(embedding, fully_triangulate=True):
    if False:
        i = 10
        return i + 15
    'Triangulates the embedding.\n\n    Traverses faces of the embedding and adds edges to a copy of the\n    embedding to triangulate it.\n    The method also ensures that the resulting graph is 2-connected by adding\n    edges if the same vertex is contained twice on a path around a face.\n\n    Parameters\n    ----------\n    embedding : nx.PlanarEmbedding\n        The input graph must contain at least 3 nodes.\n\n    fully_triangulate : bool\n        If set to False the face with the most nodes is chooses as outer face.\n        This outer face does not get triangulated.\n\n    Returns\n    -------\n    (embedding, outer_face) : (nx.PlanarEmbedding, list) tuple\n        The element `embedding` is a new embedding containing all edges from\n        the input embedding and the additional edges to triangulate the graph.\n        The element `outer_face` is a list of nodes that lie on the outer face.\n        If the graph is fully triangulated these are three arbitrary connected\n        nodes.\n\n    '
    if len(embedding.nodes) <= 1:
        return (embedding, list(embedding.nodes))
    embedding = nx.PlanarEmbedding(embedding)
    component_nodes = [next(iter(x)) for x in nx.connected_components(embedding)]
    for i in range(len(component_nodes) - 1):
        v1 = component_nodes[i]
        v2 = component_nodes[i + 1]
        embedding.connect_components(v1, v2)
    outer_face = []
    face_list = []
    edges_visited = set()
    for v in embedding.nodes():
        for w in embedding.neighbors_cw_order(v):
            new_face = make_bi_connected(embedding, v, w, edges_visited)
            if new_face:
                face_list.append(new_face)
                if len(new_face) > len(outer_face):
                    outer_face = new_face
    for face in face_list:
        if face is not outer_face or fully_triangulate:
            triangulate_face(embedding, face[0], face[1])
    if fully_triangulate:
        v1 = outer_face[0]
        v2 = outer_face[1]
        v3 = embedding[v2][v1]['ccw']
        outer_face = [v1, v2, v3]
    return (embedding, outer_face)

def make_bi_connected(embedding, starting_node, outgoing_node, edges_counted):
    if False:
        for i in range(10):
            print('nop')
    'Triangulate a face and make it 2-connected\n\n    This method also adds all edges on the face to `edges_counted`.\n\n    Parameters\n    ----------\n    embedding: nx.PlanarEmbedding\n        The embedding that defines the faces\n    starting_node : node\n        A node on the face\n    outgoing_node : node\n        A node such that the half edge (starting_node, outgoing_node) belongs\n        to the face\n    edges_counted: set\n        Set of all half-edges that belong to a face that have been visited\n\n    Returns\n    -------\n    face_nodes: list\n        A list of all nodes at the border of this face\n    '
    if (starting_node, outgoing_node) in edges_counted:
        return []
    edges_counted.add((starting_node, outgoing_node))
    v1 = starting_node
    v2 = outgoing_node
    face_list = [starting_node]
    face_set = set(face_list)
    (_, v3) = embedding.next_face_half_edge(v1, v2)
    while v2 != starting_node or v3 != outgoing_node:
        if v1 == v2:
            raise nx.NetworkXException('Invalid half-edge')
        if v2 in face_set:
            embedding.add_half_edge_cw(v1, v3, v2)
            embedding.add_half_edge_ccw(v3, v1, v2)
            edges_counted.add((v2, v3))
            edges_counted.add((v3, v1))
            v2 = v1
        else:
            face_set.add(v2)
            face_list.append(v2)
        v1 = v2
        (v2, v3) = embedding.next_face_half_edge(v2, v3)
        edges_counted.add((v1, v2))
    return face_list