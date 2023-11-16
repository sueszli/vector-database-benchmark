"""Helper functions for community-finding algorithms."""
import networkx as nx
__all__ = ['is_partition']

@nx._dispatch
def is_partition(G, communities):
    if False:
        return 10
    'Returns *True* if `communities` is a partition of the nodes of `G`.\n\n    A partition of a universe set is a family of pairwise disjoint sets\n    whose union is the entire universe set.\n\n    Parameters\n    ----------\n    G : NetworkX graph.\n\n    communities : list or iterable of sets of nodes\n        If not a list, the iterable is converted internally to a list.\n        If it is an iterator it is exhausted.\n\n    '
    if not isinstance(communities, list):
        communities = list(communities)
    nodes = {n for c in communities for n in c if n in G}
    return len(G) == len(nodes) == sum((len(c) for c in communities))