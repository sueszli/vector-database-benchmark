"""
==========
Morse Trie
==========

A prefix tree (aka a "trie") representing the Morse encoding of the alphabet.
A letter can be encoded by tracing the path from the corresponding node in the
tree to the root node, reversing the order of the symbols encountered along
the path.
"""
import networkx as nx
dot = '•'
dash = '—'
morse_direct_mapping = {'a': dot + dash, 'b': dash + dot * 3, 'c': dash + dot + dash + dot, 'd': dash + dot * 2, 'e': dot, 'f': dot * 2 + dash + dot, 'g': dash * 2 + dot, 'h': dot * 4, 'i': dot * 2, 'j': dot + dash * 3, 'k': dash + dot + dash, 'l': dot + dash + dot * 2, 'm': dash * 2, 'n': dash + dot, 'o': dash * 3, 'p': dot + dash * 2 + dot, 'q': dash * 2 + dot + dash, 'r': dot + dash + dot, 's': dot * 3, 't': dash, 'u': dot * 2 + dash, 'v': dot * 3 + dash, 'w': dot + dash * 2, 'x': dash + dot * 2 + dash, 'y': dash + dot + dash * 2, 'z': dash * 2 + dot * 2}
morse_mapping_sorted = dict(sorted(morse_direct_mapping.items(), key=lambda item: (len(item[1]), item[1])))
reverse_mapping = {v: k for (k, v) in morse_direct_mapping.items()}
reverse_mapping[''] = ''
G = nx.DiGraph()
for (node, char) in morse_mapping_sorted.items():
    pred = char[:-1]
    G.add_edge(reverse_mapping[pred], node, char=char[-1])
for (i, layer) in enumerate(nx.topological_generations(G)):
    for n in layer:
        G.nodes[n]['layer'] = i
pos = nx.multipartite_layout(G, subset_key='layer', align='horizontal')
for k in pos:
    pos[k][-1] *= -1
nx.draw(G, pos=pos, with_labels=True)
elabels = {(u, v): l for (u, v, l) in G.edges(data='char')}
nx.draw_networkx_edge_labels(G, pos, edge_labels=elabels)

def morse_encode(letter):
    if False:
        i = 10
        return i + 15
    pred = next(G.predecessors(letter))
    symbol = G[pred][letter]['char']
    if pred != '':
        return morse_encode(pred) + symbol
    return symbol
import string
for letter in string.ascii_lowercase:
    assert morse_encode(letter) == morse_direct_mapping[letter]
print(' '.join([morse_encode(ltr) for ltr in 'ilovenetworkx']))