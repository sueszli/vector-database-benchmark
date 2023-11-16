"""Asynchronous Fluid Communities algorithm for community detection."""
from collections import Counter
import networkx as nx
from networkx.algorithms.components import is_connected
from networkx.exception import NetworkXError
from networkx.utils import groups, not_implemented_for, py_random_state
__all__ = ['asyn_fluidc']

@not_implemented_for('directed', 'multigraph')
@py_random_state(3)
@nx._dispatch
def asyn_fluidc(G, k, max_iter=100, seed=None):
    if False:
        i = 10
        return i + 15
    'Returns communities in `G` as detected by Fluid Communities algorithm.\n\n    The asynchronous fluid communities algorithm is described in\n    [1]_. The algorithm is based on the simple idea of fluids interacting\n    in an environment, expanding and pushing each other. Its initialization is\n    random, so found communities may vary on different executions.\n\n    The algorithm proceeds as follows. First each of the initial k communities\n    is initialized in a random vertex in the graph. Then the algorithm iterates\n    over all vertices in a random order, updating the community of each vertex\n    based on its own community and the communities of its neighbours. This\n    process is performed several times until convergence.\n    At all times, each community has a total density of 1, which is equally\n    distributed among the vertices it contains. If a vertex changes of\n    community, vertex densities of affected communities are adjusted\n    immediately. When a complete iteration over all vertices is done, such that\n    no vertex changes the community it belongs to, the algorithm has converged\n    and returns.\n\n    This is the original version of the algorithm described in [1]_.\n    Unfortunately, it does not support weighted graphs yet.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        Graph must be simple and undirected.\n\n    k : integer\n        The number of communities to be found.\n\n    max_iter : integer\n        The number of maximum iterations allowed. By default 100.\n\n    seed : integer, random_state, or None (default)\n        Indicator of random number generation state.\n        See :ref:`Randomness<randomness>`.\n\n    Returns\n    -------\n    communities : iterable\n        Iterable of communities given as sets of nodes.\n\n    Notes\n    -----\n    k variable is not an optional argument.\n\n    References\n    ----------\n    .. [1] Parés F., Garcia-Gasulla D. et al. "Fluid Communities: A\n       Competitive and Highly Scalable Community Detection Algorithm".\n       [https://arxiv.org/pdf/1703.09307.pdf].\n    '
    if not isinstance(k, int):
        raise NetworkXError('k must be an integer.')
    if not k > 0:
        raise NetworkXError('k must be greater than 0.')
    if not is_connected(G):
        raise NetworkXError('Fluid Communities require connected Graphs.')
    if len(G) < k:
        raise NetworkXError('k cannot be bigger than the number of nodes.')
    max_density = 1.0
    vertices = list(G)
    seed.shuffle(vertices)
    communities = {n: i for (i, n) in enumerate(vertices[:k])}
    density = {}
    com_to_numvertices = {}
    for vertex in communities:
        com_to_numvertices[communities[vertex]] = 1
        density[communities[vertex]] = max_density
    iter_count = 0
    cont = True
    while cont:
        cont = False
        iter_count += 1
        vertices = list(G)
        seed.shuffle(vertices)
        for vertex in vertices:
            com_counter = Counter()
            try:
                com_counter.update({communities[vertex]: density[communities[vertex]]})
            except KeyError:
                pass
            for v in G[vertex]:
                try:
                    com_counter.update({communities[v]: density[communities[v]]})
                except KeyError:
                    continue
            new_com = -1
            if len(com_counter.keys()) > 0:
                max_freq = max(com_counter.values())
                best_communities = [com for (com, freq) in com_counter.items() if max_freq - freq < 0.0001]
                try:
                    if communities[vertex] in best_communities:
                        new_com = communities[vertex]
                except KeyError:
                    pass
                if new_com == -1:
                    cont = True
                    new_com = seed.choice(best_communities)
                    try:
                        com_to_numvertices[communities[vertex]] -= 1
                        density[communities[vertex]] = max_density / com_to_numvertices[communities[vertex]]
                    except KeyError:
                        pass
                    communities[vertex] = new_com
                    com_to_numvertices[communities[vertex]] += 1
                    density[communities[vertex]] = max_density / com_to_numvertices[communities[vertex]]
        if iter_count > max_iter:
            break
    return iter(groups(communities).values())