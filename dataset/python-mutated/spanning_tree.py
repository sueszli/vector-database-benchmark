import itertools
import math
import torch
from torch.distributions import constraints
from torch.distributions.utils import lazy_property
from pyro.distributions.torch_distribution import TorchDistribution

class SpanningTree(TorchDistribution):
    """
    Distribution over spanning trees on a fixed number ``V`` of vertices.

    A tree is represented as :class:`torch.LongTensor` ``edges`` of shape
    ``(V-1,2)`` satisfying the following properties:

    1. The edges constitute a tree, i.e. are connected and cycle free.
    2. Each edge ``(v1,v2) = edges[e]`` is sorted, i.e. ``v1 < v2``.
    3. The entire tensor is sorted in colexicographic order.

    Use :func:`validate_edges` to verify `edges` are correctly formed.

    The ``edge_logits`` tensor has one entry for each of the ``V*(V-1)//2``
    edges in the complete graph on ``V`` vertices, where edges are each sorted
    and the edge order is colexicographic::

        (0,1), (0,2), (1,2), (0,3), (1,3), (2,3), (0,4), (1,4), (2,4), ...

    This ordering corresponds to the size-independent pairing function::

        k = v1 + v2 * (v2 - 1) // 2

    where ``k`` is the rank of the edge ``(v1,v2)`` in the complete graph.
    To convert a matrix of edge logits to the linear representation used here::

        assert my_matrix.shape == (V, V)
        i, j = make_complete_graph(V)
        edge_logits = my_matrix[i, j]

    :param torch.Tensor edge_logits: A tensor of length ``V*(V-1)//2``
        containing logits (aka negative energies) of all edges in the complete
        graph on ``V`` vertices. See above comment for edge ordering.
    :param dict sampler_options: An optional dict of sampler options including:
        ``mcmc_steps`` defaulting to a single MCMC step (which is pretty good);
        ``initial_edges`` defaulting to a cheap approximate sample;
        ``backend`` one of "python" or "cpp", defaulting to "python".
    """
    arg_constraints = {'edge_logits': constraints.real}
    support = constraints.nonnegative_integer
    has_enumerate_support = True

    def __init__(self, edge_logits, sampler_options=None, validate_args=None):
        if False:
            return 10
        if edge_logits.is_cuda:
            raise NotImplementedError('SpanningTree does not support cuda tensors')
        K = len(edge_logits)
        V = int(round(0.5 + (0.25 + 2 * K) ** 0.5))
        assert K == V * (V - 1) // 2
        E = V - 1
        event_shape = (E, 2)
        batch_shape = ()
        self.edge_logits = edge_logits
        super().__init__(batch_shape, event_shape, validate_args=validate_args)
        if self._validate_args:
            if edge_logits.shape != (K,):
                raise ValueError('Expected edge_logits of shape ({},), but got shape {}'.format(K, edge_logits.shape))
        self.num_vertices = V
        self.sampler_options = {} if sampler_options is None else sampler_options

    def validate_edges(self, edges):
        if False:
            return 10
        '\n        Validates a batch of ``edges`` tensors, as returned by :meth:`sample` or\n        :meth:`enumerate_support` or as input to :meth:`log_prob()`.\n\n        :param torch.LongTensor edges: A batch of edges.\n        :raises: ValueError\n        :returns: None\n        '
        if edges.shape[-2:] != self.event_shape:
            raise ValueError('Invalid edges shape: {}'.format(edges.shape))
        if not ((0 <= edges) & (edges < self.num_vertices)).all():
            raise ValueError('Invalid vertex ids:\n{}'.format(edges))
        if not (edges[..., 0] < edges[..., 1]).all():
            raise ValueError('Vertices are not sorted in each edge:\n{}'.format(edges))
        if not ((edges[..., :-1, 1] < edges[..., 1:, 1]) | (edges[..., :-1, 1] == edges[..., 1:, 1]) & (edges[..., :-1, 0] < edges[..., 1:, 0])).all():
            raise ValueError('Edges are not sorted colexicographically:\n{}'.format(edges))
        V = self.num_vertices
        for i in itertools.product(*map(range, edges.shape[:-2])):
            edges_i = edges[i]
            connected = torch.eye(V, dtype=torch.float)
            connected[edges_i[:, 0], edges_i[:, 1]] = 1
            connected[edges_i[:, 1], edges_i[:, 0]] = 1
            for i in range(int(math.ceil(V ** 0.5))):
                connected = connected.mm(connected).clamp_(max=1)
            if not connected.min() > 0:
                raise ValueError('Edges do not constitute a tree:\n{}'.format(edges_i))

    @lazy_property
    def log_partition_function(self):
        if False:
            for i in range(10):
                print('nop')
        V = self.num_vertices
        (v1, v2) = make_complete_graph(V).unbind(0)
        logits = self.edge_logits.new_full((V, V), -math.inf)
        logits[v1, v2] = self.edge_logits
        logits[v2, v1] = self.edge_logits
        log_diag = logits.logsumexp(-1)
        shift = 0.5 * log_diag
        laplacian = torch.eye(V) - (logits - shift - shift[:, None]).exp()
        truncated = laplacian[:-1, :-1]
        try:
            import gpytorch
            log_det = gpytorch.lazy.NonLazyTensor(truncated).logdet()
        except ImportError:
            log_det = torch.linalg.cholesky(truncated).diag().log().sum() * 2
        return log_det + log_diag[:-1].sum()

    def log_prob(self, edges):
        if False:
            return 10
        if self._validate_args:
            self.validate_edges(edges)
        v1 = edges[..., 0]
        v2 = edges[..., 1]
        k = v1 + v2 * (v2 - 1) // 2
        return self.edge_logits[k].sum(-1) - self.log_partition_function

    def sample(self, sample_shape=torch.Size()):
        if False:
            while True:
                i = 10
        '\n        This sampler is implemented using MCMC run for a small number of steps\n        after being initialized by a cheap approximate sampler. This sampler is\n        approximate and cubic time. This is faster than the classic\n        Aldous-Broder sampler [1,2], especially for graphs with large mixing\n        time. Recent research [3,4] proposes samplers that run in\n        sub-matrix-multiply time but are more complex to implement.\n\n        **References**\n\n        [1] `Generating random spanning trees`\n            Andrei Broder (1989)\n        [2] `The Random Walk Construction of Uniform Spanning Trees and Uniform Labelled Trees`,\n            David J. Aldous (1990)\n        [3] `Sampling Random Spanning Trees Faster than Matrix Multiplication`,\n            David Durfee, Rasmus Kyng, John Peebles, Anup B. Rao, Sushant Sachdeva\n            (2017) https://arxiv.org/abs/1611.07451\n        [4] `An almost-linear time algorithm for uniform random spanning tree generation`,\n            Aaron Schild (2017) https://arxiv.org/abs/1711.06455\n        '
        if sample_shape:
            raise NotImplementedError('SpanningTree does not support batching')
        edges = sample_tree(self.edge_logits, **self.sampler_options)
        assert edges.dim() >= 2 and edges.shape[-2:] == self.event_shape
        return edges

    def enumerate_support(self, expand=True):
        if False:
            print('Hello World!')
        '\n        This is implemented for trees with up to 6 vertices (and 5 edges).\n        '
        trees = enumerate_spanning_trees(self.num_vertices)
        return torch.tensor(trees, dtype=torch.long)

    @property
    def mode(self):
        if False:
            i = 10
            return i + 15
        '\n        :returns: The maximum weight spanning tree.\n        :rtype: Tensor\n        '
        backend = self.sampler_options.get('backend', 'python')
        return find_best_tree(self.edge_logits, backend=backend)

    @property
    def edge_mean(self):
        if False:
            i = 10
            return i + 15
        "\n        Computes marginal probabilities of each edge being active.\n\n        .. note:: This is similar to other distributions' ``.mean()``\n            method, but with a different shape because this distribution's\n            values are not encoded as binary matrices.\n\n        :returns: A symmetric square ``(V,V)``-shaped matrix with values\n            in ``[0,1]`` denoting the marginal probability of each edge\n            being in a sampled value.\n        :rtype: Tensor\n        "
        V = self.num_vertices
        (v1, v2) = make_complete_graph(V).unbind(0)
        logits = self.edge_logits - self.edge_logits.max()
        w = self.edge_logits.new_zeros(V, V)
        w[v1, v2] = w[v2, v1] = logits.exp()
        laplacian = w.sum(-1).diag_embed() - w
        inv = (laplacian + 1 / V).pinverse()
        resistance = inv.diag() + inv.diag()[..., None] - 2 * inv
        return resistance * w
_cpp_module = None

def _get_cpp_module():
    if False:
        i = 10
        return i + 15
    '\n    JIT compiles the cpp_spanning_tree module.\n    '
    global _cpp_module
    if _cpp_module is None:
        import os
        from torch.utils.cpp_extension import load
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'spanning_tree.cpp')
        _cpp_module = load(name='cpp_spanning_tree', sources=[path], extra_cflags=['-O2'], verbose=True)
    return _cpp_module

def make_complete_graph(num_vertices, backend='python'):
    if False:
        for i in range(10):
            print('nop')
    '\n    Constructs a complete graph.\n\n    The pairing function is: ``k = v1 + v2 * (v2 - 1) // 2``\n\n    :param int num_vertices: Number of vertices.\n    :returns: a 2 x K grid of (vertex, vertex) pairs.\n    '
    if backend == 'python':
        return _make_complete_graph(num_vertices)
    elif backend == 'cpp':
        return _get_cpp_module().make_complete_graph(num_vertices)
    else:
        raise ValueError('unknown backend: {}'.format(repr(backend)))

def _make_complete_graph(num_vertices):
    if False:
        while True:
            i = 10
    if num_vertices < 2:
        raise ValueError('PyTorch cannot handle zero-sized multidimensional tensors')
    V = num_vertices
    K = V * (V - 1) // 2
    v1 = torch.arange(V)
    v2 = torch.arange(V).unsqueeze(-1)
    (v1, v2) = torch.broadcast_tensors(v1, v2)
    v1 = v1.contiguous().view(-1)
    v2 = v2.contiguous().view(-1)
    mask = v1 < v2
    grid = torch.stack((v1[mask], v2[mask]))
    assert grid.shape == (2, K)
    return grid

def _remove_edge(grid, edge_ids, neighbors, components, e):
    if False:
        i = 10
        return i + 15
    '\n    Remove an edge from a spanning tree.\n    '
    k = edge_ids[e]
    v1 = grid[0, k].item()
    v2 = grid[1, k].item()
    neighbors[v1].remove(v2)
    neighbors[v2].remove(v1)
    components[v1] = 1
    pending = [v1]
    while pending:
        v1 = pending.pop()
        for v2 in neighbors[v1]:
            if not components[v2]:
                components[v2] = 1
                pending.append(v2)
    return k

def _add_edge(grid, edge_ids, neighbors, components, e, k):
    if False:
        print('Hello World!')
    '\n    Add an edge connecting two components to create a spanning tree.\n    '
    edge_ids[e] = k
    v1 = grid[0, k].item()
    v2 = grid[1, k].item()
    neighbors[v1].add(v2)
    neighbors[v2].add(v1)
    components.fill_(0)

def _find_valid_edges(components, valid_edge_ids):
    if False:
        i = 10
        return i + 15
    '\n    Find all edges between two components in a complete undirected graph.\n\n    :param components: A [V]-shaped array of boolean component ids. This\n        assumes there are exactly two nonemtpy components.\n    :param valid_edge_ids: An uninitialized array where output is written. On\n        return, the subarray valid_edge_ids[:end] will contain edge ids k for all\n        valid edges.\n    :returns: The number of valid edges found.\n    '
    k = 0
    end = 0
    for (v2, c2) in enumerate(components):
        for v1 in range(v2):
            if c2 ^ components[v1]:
                valid_edge_ids[end] = k
                end += 1
            k += 1
    return end

@torch.no_grad()
def _sample_tree_mcmc(edge_logits, edges):
    if False:
        i = 10
        return i + 15
    if len(edges) <= 1:
        return edges
    E = len(edges)
    V = E + 1
    K = V * (V - 1) // 2
    grid = make_complete_graph(V)
    edge_ids = torch.empty(E, dtype=torch.long)
    neighbors = {v: set() for v in range(V)}
    components = torch.zeros(V, dtype=torch.bool)
    for e in range(E):
        (v1, v2) = map(int, edges[e])
        assert v1 < v2
        edge_ids[e] = v1 + v2 * (v2 - 1) // 2
        neighbors[v1].add(v2)
        neighbors[v2].add(v1)
    valid_edges_buffer = torch.empty(K, dtype=torch.long)
    for e in torch.randperm(E):
        e = int(e)
        k = _remove_edge(grid, edge_ids, neighbors, components, e)
        num_valid_edges = _find_valid_edges(components, valid_edges_buffer)
        valid_edge_ids = valid_edges_buffer[:num_valid_edges]
        valid_logits = edge_logits[valid_edge_ids]
        valid_probs = (valid_logits - valid_logits.max()).exp()
        total_prob = valid_probs.sum()
        if total_prob > 0:
            sample = torch.multinomial(valid_probs, 1)[0]
            k = valid_edge_ids[sample]
        _add_edge(grid, edge_ids, neighbors, components, e, k)
    edge_ids = edge_ids.sort()[0]
    edges = torch.empty((E, 2), dtype=torch.long)
    edges[:, 0] = grid[0, edge_ids]
    edges[:, 1] = grid[1, edge_ids]
    return edges

def sample_tree_mcmc(edge_logits, edges, backend='python'):
    if False:
        i = 10
        return i + 15
    '\n    Sample a random spanning tree of a dense weighted graph using MCMC.\n\n    This uses Gibbs sampling on edges. Consider E undirected edges that can\n    move around a graph of ``V=1+E`` vertices. The edges are constrained so\n    that no two edges can span the same pair of vertices and so that the edges\n    must form a spanning tree. To Gibbs sample, chose one of the E edges at\n    random and move it anywhere else in the graph. After we remove the edge,\n    notice that the graph is split into two connected components. The\n    constraints imply that the edge must be replaced so as to connect the two\n    components.  Hence to Gibbs sample, we collect all such bridging\n    (vertex,vertex) pairs and sample from them in proportion to\n    ``exp(edge_logits)``.\n\n    :param torch.Tensor edge_logits: A length-K array of nonnormalized log\n        probabilities.\n    :param torch.Tensor edges: An E x 2 tensor of initial edges in the form\n        of (vertex,vertex) pairs. Each edge should be sorted and the entire\n        tensor should be lexicographically sorted.\n    :returns: An E x 2 tensor of edges in the form of (vertex,vertex) pairs.\n        Each edge should be sorted and the entire tensor should be\n        lexicographically sorted.\n    :rtype: torch.Tensor\n    '
    if backend == 'python':
        return _sample_tree_mcmc(edge_logits, edges)
    elif backend == 'cpp':
        return _get_cpp_module().sample_tree_mcmc(edge_logits, edges)
    else:
        raise ValueError('unknown backend: {}'.format(repr(backend)))

@torch.no_grad()
def _sample_tree_approx(edge_logits):
    if False:
        i = 10
        return i + 15
    K = len(edge_logits)
    V = int(round(0.5 + (0.25 + 2 * K) ** 0.5))
    assert K == V * (V - 1) // 2
    E = V - 1
    grid = make_complete_graph(V)
    edge_ids = torch.empty((E,), dtype=torch.long)
    components = torch.zeros(V, dtype=torch.bool)
    probs = (edge_logits - edge_logits.max()).exp()
    k = torch.multinomial(probs, 1)[0]
    components[grid[:, k]] = 1
    edge_ids[0] = k
    for e in range(1, E):
        (c1, c2) = components[grid]
        mask = c1 != c2
        valid_logits = edge_logits[mask]
        probs = (valid_logits - valid_logits.max()).exp()
        k = mask.nonzero(as_tuple=False)[torch.multinomial(probs, 1)[0]]
        components[grid[:, k]] = 1
        edge_ids[e] = k
    edge_ids = edge_ids.sort()[0]
    edges = torch.empty((E, 2), dtype=torch.long)
    edges[:, 0] = grid[0, edge_ids]
    edges[:, 1] = grid[1, edge_ids]
    return edges

def sample_tree_approx(edge_logits, backend='python'):
    if False:
        while True:
            i = 10
    '\n    Approximately sample a random spanning tree of a dense weighted graph.\n\n    This is mainly useful for initializing an MCMC sampler.\n\n    :param torch.Tensor edge_logits: A length-K array of nonnormalized log\n        probabilities.\n    :returns: An E x 2 tensor of edges in the form of (vertex,vertex) pairs.\n        Each edge should be sorted and the entire tensor should be\n        lexicographically sorted.\n    :rtype: torch.Tensor\n    '
    if backend == 'python':
        return _sample_tree_approx(edge_logits)
    elif backend == 'cpp':
        return _get_cpp_module().sample_tree_approx(edge_logits)
    else:
        raise ValueError('unknown backend: {}'.format(repr(backend)))

def sample_tree(edge_logits, init_edges=None, mcmc_steps=1, backend='python'):
    if False:
        while True:
            i = 10
    edges = init_edges
    if edges is None:
        edges = sample_tree_approx(edge_logits, backend=backend)
    for step in range(mcmc_steps):
        edges = sample_tree_mcmc(edge_logits, edges, backend=backend)
    return edges

@torch.no_grad()
def _find_best_tree(edge_logits):
    if False:
        while True:
            i = 10
    K = len(edge_logits)
    V = int(round(0.5 + (0.25 + 2 * K) ** 0.5))
    assert K == V * (V - 1) // 2
    E = V - 1
    grid = make_complete_graph(V)
    edge_ids = torch.empty((E,), dtype=torch.long)
    components = torch.zeros(V, dtype=torch.bool)
    k = edge_logits.argmax(0).item()
    components[grid[:, k]] = 1
    edge_ids[0] = k
    for e in range(1, E):
        (c1, c2) = components[grid]
        mask = c1 != c2
        valid_logits = edge_logits[mask]
        k = valid_logits.argmax(0).item()
        k = mask.nonzero(as_tuple=False)[k]
        components[grid[:, k]] = 1
        edge_ids[e] = k
    edge_ids = edge_ids.sort()[0]
    edges = torch.empty((E, 2), dtype=torch.long)
    edges[:, 0] = grid[0, edge_ids]
    edges[:, 1] = grid[1, edge_ids]
    return edges

def find_best_tree(edge_logits, backend='python'):
    if False:
        while True:
            i = 10
    '\n    Find the maximum weight spanning tree of a dense weighted graph.\n\n    :param torch.Tensor edge_logits: A length-K array of nonnormalized log\n        probabilities.\n    :returns: An E x 2 tensor of edges in the form of (vertex,vertex) pairs.\n        Each edge should be sorted and the entire tensor should be\n        lexicographically sorted.\n    :rtype: torch.Tensor\n    '
    if backend == 'python':
        return _find_best_tree(edge_logits)
    elif backend == 'cpp':
        return _get_cpp_module().find_best_tree(edge_logits)
    else:
        raise ValueError('unknown backend: {}'.format(repr(backend)))
NUM_SPANNING_TREES = [1, 1, 1, 3, 16, 125, 1296, 16807, 262144, 4782969, 100000000, 2357947691, 61917364224, 1792160394037, 56693912375296, 1946195068359375, 72057594037927936, 2862423051509815793, 121439531096594251776, 5480386857784802185939]
_TREE_GENERATORS = [[[]], [[]], [[(0, 1)]], [[(0, 1), (0, 2)]], [[(0, 1), (0, 2), (0, 3)], [(0, 1), (1, 2), (2, 3)]], [[(0, 1), (0, 2), (0, 3), (0, 4)], [(0, 1), (0, 2), (0, 3), (1, 4)], [(0, 1), (1, 2), (2, 3), (3, 4)]], [[(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)], [(0, 1), (0, 2), (0, 3), (0, 4), (1, 5)], [(0, 1), (0, 2), (0, 3), (1, 4), (4, 5)], [(0, 1), (0, 2), (0, 3), (2, 4), (3, 5)], [(0, 1), (0, 2), (0, 3), (3, 4), (3, 5)], [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]]]

def _permute_tree(perm, tree):
    if False:
        i = 10
        return i + 15
    edges = [tuple(sorted([perm[u], perm[v]])) for (u, v) in tree]
    edges.sort(key=lambda uv: (uv[1], uv[0]))
    return tuple(edges)

def _close_under_permutations(V, tree_generators):
    if False:
        for i in range(10):
            print('nop')
    vertices = list(range(V))
    trees = []
    for tree in tree_generators:
        trees.extend(set((_permute_tree(perm, tree) for perm in itertools.permutations(vertices))))
    trees.sort()
    return trees

def enumerate_spanning_trees(V):
    if False:
        while True:
            i = 10
    '\n    Compute the set of spanning trees on V vertices.\n    '
    if V >= len(_TREE_GENERATORS):
        raise NotImplementedError('enumerate_spanning_trees() is implemented only for trees with up to {} vertices'.format(len(_TREE_GENERATORS) - 1))
    all_trees = _close_under_permutations(V, _TREE_GENERATORS[V])
    assert len(all_trees) == NUM_SPANNING_TREES[V]
    return all_trees