"""
Circular Layout
===============

This module contains several graph layouts which rely heavily on circles.
"""
import numpy as np
from ..util import _straight_line_vertices, issparse

def circular(adjacency_mat, directed=False):
    if False:
        print('Hello World!')
    'Places all nodes on a single circle.\n\n    Parameters\n    ----------\n    adjacency_mat : matrix or sparse\n        The graph adjacency matrix\n    directed : bool\n        Whether the graph is directed. If this is True, is will also\n        generate the vertices for arrows, which can be passed to an\n        ArrowVisual.\n\n    Yields\n    ------\n    (node_vertices, line_vertices, arrow_vertices) : tuple\n        Yields the node and line vertices in a tuple. This layout only yields a\n        single time, and has no builtin animation\n    '
    if issparse(adjacency_mat):
        adjacency_mat = adjacency_mat.tocoo()
    num_nodes = adjacency_mat.shape[0]
    t = np.linspace(0, 2 * np.pi, num_nodes, endpoint=False)
    node_coords = (0.5 * np.array([np.cos(t), np.sin(t)]) + 0.5).T
    node_coords = node_coords.astype(np.float32)
    (line_vertices, arrows) = _straight_line_vertices(adjacency_mat, node_coords, directed)
    yield (node_coords, line_vertices, arrows)