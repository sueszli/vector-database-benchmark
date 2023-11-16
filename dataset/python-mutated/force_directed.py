"""
Force-Directed Graph Layout
===========================

This module contains implementations for a force-directed layout, where the
graph is modelled like a collection of springs or as a collection of
particles attracting and repelling each other. The whole graph tries to
reach a state which requires the minimum energy.
"""
import numpy as np
try:
    from scipy.sparse import issparse
except ImportError:

    def issparse(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return False
from ..util import _straight_line_vertices, _rescale_layout

class fruchterman_reingold(object):
    """Fruchterman-Reingold implementation adapted from NetworkX.

    In the Fruchterman-Reingold algorithm, the whole graph is modelled as a
    collection of particles, it runs a simplified particle simulation to
    find a nice layout for the graph.

    Parameters
    ----------
    optimal : number
        Optimal distance between nodes. Defaults to :math:`1/\\\\sqrt{N}` where
        N is the number of nodes.
    iterations : int
        Number of iterations to perform for layout calculation.
    pos : array
        Initial positions of the nodes

    Notes
    -----
    The algorithm is explained in more detail in the original paper [1]_.

    .. [1] Fruchterman, Thomas MJ, and Edward M. Reingold. "Graph drawing by
       force-directed placement." Softw., Pract. Exper. 21.11 (1991),
       1129-1164.
    """

    def __init__(self, optimal=None, iterations=50, pos=None):
        if False:
            i = 10
            return i + 15
        self.dim = 2
        self.optimal = optimal
        self.iterations = iterations
        self.num_nodes = None
        self.pos = pos

    def __call__(self, adjacency_mat, directed=False):
        if False:
            return 10
        '\n        Starts the calculation of the graph layout.\n\n        This is a generator, and after each iteration it yields the new\n        positions for the nodes, together with the vertices for the edges\n        and the arrows.\n\n        There are two solvers here: one specially adapted for SciPy sparse\n        matrices, and the other for larger networks.\n\n        Parameters\n        ----------\n        adjacency_mat : array\n            The graph adjacency matrix.\n        directed : bool\n            Wether the graph is directed or not. If this is True,\n            it will draw arrows for directed edges.\n\n        Yields\n        ------\n        layout : tuple\n            For each iteration of the layout calculation it yields a tuple\n            containing (node_vertices, line_vertices, arrow_vertices). These\n            vertices can be passed to the `MarkersVisual` and `ArrowVisual`.\n        '
        if adjacency_mat.shape[0] != adjacency_mat.shape[1]:
            raise ValueError('Adjacency matrix should be square.')
        self.num_nodes = adjacency_mat.shape[0]
        if issparse(adjacency_mat):
            solver = self._sparse_fruchterman_reingold
        else:
            solver = self._fruchterman_reingold
        for result in solver(adjacency_mat, directed):
            yield result

    def _fruchterman_reingold(self, adjacency_mat, directed=False):
        if False:
            while True:
                i = 10
        if self.optimal is None:
            self.optimal = 1 / np.sqrt(self.num_nodes)
        if self.pos is None:
            pos = np.asarray(np.random.random((self.num_nodes, self.dim)), dtype=np.float32)
        else:
            pos = self.pos.astype(np.float32)
        (line_vertices, arrows) = _straight_line_vertices(adjacency_mat, pos, directed)
        yield (pos, line_vertices, arrows)
        t = 0.1
        dt = t / float(self.iterations + 1)
        for iteration in range(self.iterations):
            delta_pos = _calculate_delta_pos(adjacency_mat, pos, t, self.optimal)
            pos += delta_pos
            _rescale_layout(pos)
            t -= dt
            (line_vertices, arrows) = _straight_line_vertices(adjacency_mat, pos, directed)
            yield (pos, line_vertices, arrows)

    def _sparse_fruchterman_reingold(self, adjacency_mat, directed=False):
        if False:
            i = 10
            return i + 15
        if self.optimal is None:
            self.optimal = 1 / np.sqrt(self.num_nodes)
        adjacency_arr = adjacency_mat.toarray()
        adjacency_coo = adjacency_mat.tocoo()
        if self.pos is None:
            pos = np.asarray(np.random.random((self.num_nodes, self.dim)), dtype=np.float32)
        else:
            pos = self.pos.astype(np.float32)
        (line_vertices, arrows) = _straight_line_vertices(adjacency_coo, pos, directed)
        yield (pos, line_vertices, arrows)
        t = 0.1
        dt = t / float(self.iterations + 1)
        for iteration in range(self.iterations):
            delta_pos = _calculate_delta_pos(adjacency_arr, pos, t, self.optimal)
            pos += delta_pos
            _rescale_layout(pos)
            t -= dt
            (line_vertices, arrows) = _straight_line_vertices(adjacency_coo, pos, directed)
            yield (pos, line_vertices, arrows)

def _calculate_delta_pos(adjacency_arr, pos, t, optimal):
    if False:
        while True:
            i = 10
    'Helper to calculate the delta position'
    delta = pos[:, np.newaxis, :] - pos
    distance2 = (delta * delta).sum(axis=-1)
    distance2 = np.where(distance2 < 0.0001, 0.0001, distance2)
    distance = np.sqrt(distance2)
    displacement = np.zeros((len(delta), 2))
    for ii in range(2):
        displacement[:, ii] = (delta[:, :, ii] * (optimal * optimal / (distance * distance) - adjacency_arr * distance / optimal)).sum(axis=1)
    length = np.sqrt((displacement ** 2).sum(axis=1))
    length = np.where(length < 0.01, 0.1, length)
    delta_pos = displacement * t / length[:, np.newaxis]
    return delta_pos