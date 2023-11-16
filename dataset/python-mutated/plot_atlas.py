"""
=====
Atlas
=====

Atlas of all connected graphs with up to 6 nodes.

This example uses Graphviz via PyGraphviz.

The image should show 142 graphs.
We don't plot the empty graph nor the single node graph.
(142 is the sum of values 2 to n=6 in sequence oeis.org/A001349).
"""
import random
import matplotlib.pyplot as plt
import networkx as nx
GraphMatcher = nx.isomorphism.vf2userfunc.GraphMatcher

def atlas6():
    if False:
        print('Hello World!')
    'Return the atlas of all connected graphs with at most 6 nodes'
    Atlas = nx.graph_atlas_g()[3:209]
    U = nx.Graph()
    for G in Atlas:
        if nx.number_connected_components(G) == 1:
            if not GraphMatcher(U, G).subgraph_is_isomorphic():
                U = nx.disjoint_union(U, G)
    return U
G = atlas6()
print(G)
print(nx.number_connected_components(G), 'connected components')
plt.figure(1, figsize=(8, 8))
pos = nx.nx_agraph.graphviz_layout(G, prog='neato')
C = (G.subgraph(c) for c in nx.connected_components(G))
for g in C:
    c = [random.random()] * nx.number_of_nodes(g)
    nx.draw(g, pos, node_size=40, node_color=c, vmin=0.0, vmax=1.0, with_labels=False)
plt.show()