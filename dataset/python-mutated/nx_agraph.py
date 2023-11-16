"""
***************
Graphviz AGraph
***************

Interface to pygraphviz AGraph class.

Examples
--------
>>> G = nx.complete_graph(5)
>>> A = nx.nx_agraph.to_agraph(G)
>>> H = nx.nx_agraph.from_agraph(A)

See Also
--------
 - Pygraphviz: http://pygraphviz.github.io/
 - Graphviz:      https://www.graphviz.org
 - DOT Language:  http://www.graphviz.org/doc/info/lang.html
"""
import os
import tempfile
import networkx as nx
__all__ = ['from_agraph', 'to_agraph', 'write_dot', 'read_dot', 'graphviz_layout', 'pygraphviz_layout', 'view_pygraphviz']

@nx._dispatch(graphs=None)
def from_agraph(A, create_using=None):
    if False:
        i = 10
        return i + 15
    'Returns a NetworkX Graph or DiGraph from a PyGraphviz graph.\n\n    Parameters\n    ----------\n    A : PyGraphviz AGraph\n      A graph created with PyGraphviz\n\n    create_using : NetworkX graph constructor, optional (default=None)\n       Graph type to create. If graph instance, then cleared before populated.\n       If `None`, then the appropriate Graph type is inferred from `A`.\n\n    Examples\n    --------\n    >>> K5 = nx.complete_graph(5)\n    >>> A = nx.nx_agraph.to_agraph(K5)\n    >>> G = nx.nx_agraph.from_agraph(A)\n\n    Notes\n    -----\n    The Graph G will have a dictionary G.graph_attr containing\n    the default graphviz attributes for graphs, nodes and edges.\n\n    Default node attributes will be in the dictionary G.node_attr\n    which is keyed by node.\n\n    Edge attributes will be returned as edge data in G.  With\n    edge_attr=False the edge data will be the Graphviz edge weight\n    attribute or the value 1 if no edge weight attribute is found.\n\n    '
    if create_using is None:
        if A.is_directed():
            if A.is_strict():
                create_using = nx.DiGraph
            else:
                create_using = nx.MultiDiGraph
        elif A.is_strict():
            create_using = nx.Graph
        else:
            create_using = nx.MultiGraph
    N = nx.empty_graph(0, create_using)
    if A.name is not None:
        N.name = A.name
    N.graph.update(A.graph_attr)
    for n in A.nodes():
        str_attr = {str(k): v for (k, v) in n.attr.items()}
        N.add_node(str(n), **str_attr)
    for e in A.edges():
        (u, v) = (str(e[0]), str(e[1]))
        attr = dict(e.attr)
        str_attr = {str(k): v for (k, v) in attr.items()}
        if not N.is_multigraph():
            if e.name is not None:
                str_attr['key'] = e.name
            N.add_edge(u, v, **str_attr)
        else:
            N.add_edge(u, v, key=e.name, **str_attr)
    N.graph['graph'] = dict(A.graph_attr)
    N.graph['node'] = dict(A.node_attr)
    N.graph['edge'] = dict(A.edge_attr)
    return N

def to_agraph(N):
    if False:
        for i in range(10):
            print('nop')
    'Returns a pygraphviz graph from a NetworkX graph N.\n\n    Parameters\n    ----------\n    N : NetworkX graph\n      A graph created with NetworkX\n\n    Examples\n    --------\n    >>> K5 = nx.complete_graph(5)\n    >>> A = nx.nx_agraph.to_agraph(K5)\n\n    Notes\n    -----\n    If N has an dict N.graph_attr an attempt will be made first\n    to copy properties attached to the graph (see from_agraph)\n    and then updated with the calling arguments if any.\n\n    '
    try:
        import pygraphviz
    except ImportError as err:
        raise ImportError('requires pygraphviz http://pygraphviz.github.io/') from err
    directed = N.is_directed()
    strict = nx.number_of_selfloops(N) == 0 and (not N.is_multigraph())
    for node in N:
        if 'pos' in N.nodes[node]:
            N.nodes[node]['pos'] = '{},{}!'.format(N.nodes[node]['pos'][0], N.nodes[node]['pos'][1])
    A = pygraphviz.AGraph(name=N.name, strict=strict, directed=directed)
    A.graph_attr.update(N.graph.get('graph', {}))
    A.node_attr.update(N.graph.get('node', {}))
    A.edge_attr.update(N.graph.get('edge', {}))
    A.graph_attr.update(((k, v) for (k, v) in N.graph.items() if k not in ('graph', 'node', 'edge')))
    for (n, nodedata) in N.nodes(data=True):
        A.add_node(n)
        a = A.get_node(n)
        a.attr.update({k: str(v) for (k, v) in nodedata.items()})
    if N.is_multigraph():
        for (u, v, key, edgedata) in N.edges(data=True, keys=True):
            str_edgedata = {k: str(v) for (k, v) in edgedata.items() if k != 'key'}
            A.add_edge(u, v, key=str(key))
            a = A.get_edge(u, v)
            a.attr.update(str_edgedata)
    else:
        for (u, v, edgedata) in N.edges(data=True):
            str_edgedata = {k: str(v) for (k, v) in edgedata.items()}
            A.add_edge(u, v)
            a = A.get_edge(u, v)
            a.attr.update(str_edgedata)
    return A

def write_dot(G, path):
    if False:
        i = 10
        return i + 15
    'Write NetworkX graph G to Graphviz dot format on path.\n\n    Parameters\n    ----------\n    G : graph\n       A networkx graph\n    path : filename\n       Filename or file handle to write\n\n    Notes\n    -----\n    To use a specific graph layout, call ``A.layout`` prior to `write_dot`.\n    Note that some graphviz layouts are not guaranteed to be deterministic,\n    see https://gitlab.com/graphviz/graphviz/-/issues/1767 for more info.\n    '
    A = to_agraph(G)
    A.write(path)
    A.clear()
    return

@nx._dispatch(name='agraph_read_dot', graphs=None)
def read_dot(path):
    if False:
        for i in range(10):
            print('nop')
    'Returns a NetworkX graph from a dot file on path.\n\n    Parameters\n    ----------\n    path : file or string\n       File name or file handle to read.\n    '
    try:
        import pygraphviz
    except ImportError as err:
        raise ImportError('read_dot() requires pygraphviz http://pygraphviz.github.io/') from err
    A = pygraphviz.AGraph(file=path)
    gr = from_agraph(A)
    A.clear()
    return gr

def graphviz_layout(G, prog='neato', root=None, args=''):
    if False:
        print('Hello World!')
    'Create node positions for G using Graphviz.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n      A graph created with NetworkX\n    prog : string\n      Name of Graphviz layout program\n    root : string, optional\n      Root node for twopi layout\n    args : string, optional\n      Extra arguments to Graphviz layout program\n\n    Returns\n    -------\n    Dictionary of x, y, positions keyed by node.\n\n    Examples\n    --------\n    >>> G = nx.petersen_graph()\n    >>> pos = nx.nx_agraph.graphviz_layout(G)\n    >>> pos = nx.nx_agraph.graphviz_layout(G, prog="dot")\n\n    Notes\n    -----\n    This is a wrapper for pygraphviz_layout.\n\n    Note that some graphviz layouts are not guaranteed to be deterministic,\n    see https://gitlab.com/graphviz/graphviz/-/issues/1767 for more info.\n    '
    return pygraphviz_layout(G, prog=prog, root=root, args=args)

def pygraphviz_layout(G, prog='neato', root=None, args=''):
    if False:
        print('Hello World!')
    'Create node positions for G using Graphviz.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n      A graph created with NetworkX\n    prog : string\n      Name of Graphviz layout program\n    root : string, optional\n      Root node for twopi layout\n    args : string, optional\n      Extra arguments to Graphviz layout program\n\n    Returns\n    -------\n    node_pos : dict\n      Dictionary of x, y, positions keyed by node.\n\n    Examples\n    --------\n    >>> G = nx.petersen_graph()\n    >>> pos = nx.nx_agraph.graphviz_layout(G)\n    >>> pos = nx.nx_agraph.graphviz_layout(G, prog="dot")\n\n    Notes\n    -----\n    If you use complex node objects, they may have the same string\n    representation and GraphViz could treat them as the same node.\n    The layout may assign both nodes a single location. See Issue #1568\n    If this occurs in your case, consider relabeling the nodes just\n    for the layout computation using something similar to::\n\n        >>> H = nx.convert_node_labels_to_integers(G, label_attribute="node_label")\n        >>> H_layout = nx.nx_agraph.pygraphviz_layout(G, prog="dot")\n        >>> G_layout = {H.nodes[n]["node_label"]: p for n, p in H_layout.items()}\n\n    Note that some graphviz layouts are not guaranteed to be deterministic,\n    see https://gitlab.com/graphviz/graphviz/-/issues/1767 for more info.\n    '
    try:
        import pygraphviz
    except ImportError as err:
        raise ImportError('requires pygraphviz http://pygraphviz.github.io/') from err
    if root is not None:
        args += f'-Groot={root}'
    A = to_agraph(G)
    A.layout(prog=prog, args=args)
    node_pos = {}
    for n in G:
        node = pygraphviz.Node(A, n)
        try:
            xs = node.attr['pos'].split(',')
            node_pos[n] = tuple((float(x) for x in xs))
        except:
            print('no position for node', n)
            node_pos[n] = (0.0, 0.0)
    return node_pos

@nx.utils.open_file(5, 'w+b')
def view_pygraphviz(G, edgelabel=None, prog='dot', args='', suffix='', path=None, show=True):
    if False:
        while True:
            i = 10
    'Views the graph G using the specified layout algorithm.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n        The machine to draw.\n    edgelabel : str, callable, None\n        If a string, then it specifies the edge attribute to be displayed\n        on the edge labels. If a callable, then it is called for each\n        edge and it should return the string to be displayed on the edges.\n        The function signature of `edgelabel` should be edgelabel(data),\n        where `data` is the edge attribute dictionary.\n    prog : string\n        Name of Graphviz layout program.\n    args : str\n        Additional arguments to pass to the Graphviz layout program.\n    suffix : str\n        If `filename` is None, we save to a temporary file.  The value of\n        `suffix` will appear at the tail end of the temporary filename.\n    path : str, None\n        The filename used to save the image.  If None, save to a temporary\n        file.  File formats are the same as those from pygraphviz.agraph.draw.\n    show : bool, default = True\n        Whether to display the graph with :mod:`PIL.Image.show`,\n        default is `True`. If `False`, the rendered graph is still available\n        at `path`.\n\n    Returns\n    -------\n    path : str\n        The filename of the generated image.\n    A : PyGraphviz graph\n        The PyGraphviz graph instance used to generate the image.\n\n    Notes\n    -----\n    If this function is called in succession too quickly, sometimes the\n    image is not displayed. So you might consider time.sleep(.5) between\n    calls if you experience problems.\n\n    Note that some graphviz layouts are not guaranteed to be deterministic,\n    see https://gitlab.com/graphviz/graphviz/-/issues/1767 for more info.\n\n    '
    if not len(G):
        raise nx.NetworkXException('An empty graph cannot be drawn.')
    attrs = ['edge', 'node', 'graph']
    for attr in attrs:
        if attr not in G.graph:
            G.graph[attr] = {}
    edge_attrs = {'fontsize': '10'}
    node_attrs = {'style': 'filled', 'fillcolor': '#0000FF40', 'height': '0.75', 'width': '0.75', 'shape': 'circle'}
    graph_attrs = {}

    def update_attrs(which, attrs):
        if False:
            print('Hello World!')
        added = []
        for (k, v) in attrs.items():
            if k not in G.graph[which]:
                G.graph[which][k] = v
                added.append(k)

    def clean_attrs(which, added):
        if False:
            for i in range(10):
                print('nop')
        for attr in added:
            del G.graph[which][attr]
        if not G.graph[which]:
            del G.graph[which]
    update_attrs('edge', edge_attrs)
    update_attrs('node', node_attrs)
    update_attrs('graph', graph_attrs)
    A = to_agraph(G)
    clean_attrs('edge', edge_attrs)
    clean_attrs('node', node_attrs)
    clean_attrs('graph', graph_attrs)
    if edgelabel is not None:
        if not callable(edgelabel):

            def func(data):
                if False:
                    for i in range(10):
                        print('nop')
                return ''.join(['  ', str(data[edgelabel]), '  '])
        else:
            func = edgelabel
        if G.is_multigraph():
            for (u, v, key, data) in G.edges(keys=True, data=True):
                edge = A.get_edge(u, v, str(key))
                edge.attr['label'] = str(func(data))
        else:
            for (u, v, data) in G.edges(data=True):
                edge = A.get_edge(u, v)
                edge.attr['label'] = str(func(data))
    if path is None:
        ext = 'png'
        if suffix:
            suffix = f'_{suffix}.{ext}'
        else:
            suffix = f'.{ext}'
        path = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    else:
        pass
    A.draw(path=path, format=None, prog=prog, args=args)
    path.close()
    if show:
        from PIL import Image
        Image.open(path.name).show()
    return (path.name, A)