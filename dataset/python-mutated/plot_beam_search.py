"""
===========
Beam Search
===========

Beam search with dynamic beam width.

The progressive widening beam search repeatedly executes a beam search
with increasing beam width until the target node is found.
"""
import math
import matplotlib.pyplot as plt
import networkx as nx

def progressive_widening_search(G, source, value, condition, initial_width=1):
    if False:
        for i in range(10):
            print('nop')
    'Progressive widening beam search to find a node.\n\n    The progressive widening beam search involves a repeated beam\n    search, starting with a small beam width then extending to\n    progressively larger beam widths if the target node is not\n    found. This implementation simply returns the first node found that\n    matches the termination condition.\n\n    `G` is a NetworkX graph.\n\n    `source` is a node in the graph. The search for the node of interest\n    begins here and extends only to those nodes in the (weakly)\n    connected component of this node.\n\n    `value` is a function that returns a real number indicating how good\n    a potential neighbor node is when deciding which neighbor nodes to\n    enqueue in the breadth-first search. Only the best nodes within the\n    current beam width will be enqueued at each step.\n\n    `condition` is the termination condition for the search. This is a\n    function that takes a node as input and return a Boolean indicating\n    whether the node is the target. If no node matches the termination\n    condition, this function raises :exc:`NodeNotFound`.\n\n    `initial_width` is the starting beam width for the beam search (the\n    default is one). If no node matching the `condition` is found with\n    this beam width, the beam search is restarted from the `source` node\n    with a beam width that is twice as large (so the beam width\n    increases exponentially). The search terminates after the beam width\n    exceeds the number of nodes in the graph.\n\n    '
    if condition(source):
        return source
    log_m = math.ceil(math.log2(len(G)))
    for i in range(log_m):
        width = initial_width * pow(2, i)
        for (u, v) in nx.bfs_beam_edges(G, source, value, width):
            if condition(v):
                return v
    raise nx.NodeNotFound('no node satisfied the termination condition')
seed = 89
G = nx.gnp_random_graph(100, 0.5, seed=seed)
centrality = nx.eigenvector_centrality(G)
avg_centrality = sum(centrality.values()) / len(G)

def has_high_centrality(v):
    if False:
        print('Hello World!')
    return centrality[v] >= avg_centrality
source = 0
value = centrality.get
condition = has_high_centrality
found_node = progressive_widening_search(G, source, value, condition)
c = centrality[found_node]
print(f'found node {found_node} with centrality {c}')
pos = nx.spring_layout(G, seed=seed)
options = {'node_color': 'blue', 'node_size': 20, 'edge_color': 'grey', 'linewidths': 0, 'width': 0.1}
nx.draw(G, pos, **options)
nx.draw_networkx_nodes(G, pos, nodelist=[found_node], node_size=100, node_color='r')
plt.show()