"""
*****
Pydot
*****

Import and export NetworkX graphs in Graphviz dot format using pydot.

Either this module or nx_agraph can be used to interface with graphviz.

Examples
--------
>>> G = nx.complete_graph(5)
>>> PG = nx.nx_pydot.to_pydot(G)
>>> H = nx.nx_pydot.from_pydot(PG)

See Also
--------
 - pydot:         https://github.com/erocarrera/pydot
 - Graphviz:      https://www.graphviz.org
 - DOT Language:  http://www.graphviz.org/doc/info/lang.html
"""
import warnings
from locale import getpreferredencoding
import networkx as nx
from networkx.utils import open_file
__all__ = ['write_dot', 'read_dot', 'graphviz_layout', 'pydot_layout', 'to_pydot', 'from_pydot']

@open_file(1, mode='w')
def write_dot(G, path):
    if False:
        while True:
            i = 10
    'Write NetworkX graph G to Graphviz dot format on path.\n\n    Path can be a string or a file handle.\n    '
    msg = 'nx.nx_pydot.write_dot depends on the pydot package, which has known issues and is not actively maintained. Consider using nx.nx_agraph.write_dot instead.\n\nSee https://github.com/networkx/networkx/issues/5723'
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    P = to_pydot(G)
    path.write(P.to_string())
    return

@open_file(0, mode='r')
@nx._dispatch(name='pydot_read_dot', graphs=None)
def read_dot(path):
    if False:
        while True:
            i = 10
    'Returns a NetworkX :class:`MultiGraph` or :class:`MultiDiGraph` from the\n    dot file with the passed path.\n\n    If this file contains multiple graphs, only the first such graph is\n    returned. All graphs _except_ the first are silently ignored.\n\n    Parameters\n    ----------\n    path : str or file\n        Filename or file handle.\n\n    Returns\n    -------\n    G : MultiGraph or MultiDiGraph\n        A :class:`MultiGraph` or :class:`MultiDiGraph`.\n\n    Notes\n    -----\n    Use `G = nx.Graph(nx.nx_pydot.read_dot(path))` to return a :class:`Graph` instead of a\n    :class:`MultiGraph`.\n    '
    import pydot
    msg = 'nx.nx_pydot.read_dot depends on the pydot package, which has known issues and is not actively maintained. Consider using nx.nx_agraph.read_dot instead.\n\nSee https://github.com/networkx/networkx/issues/5723'
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    data = path.read()
    P_list = pydot.graph_from_dot_data(data)
    return from_pydot(P_list[0])

@nx._dispatch(graphs=None)
def from_pydot(P):
    if False:
        print('Hello World!')
    'Returns a NetworkX graph from a Pydot graph.\n\n    Parameters\n    ----------\n    P : Pydot graph\n      A graph created with Pydot\n\n    Returns\n    -------\n    G : NetworkX multigraph\n        A MultiGraph or MultiDiGraph.\n\n    Examples\n    --------\n    >>> K5 = nx.complete_graph(5)\n    >>> A = nx.nx_pydot.to_pydot(K5)\n    >>> G = nx.nx_pydot.from_pydot(A)  # return MultiGraph\n\n    # make a Graph instead of MultiGraph\n    >>> G = nx.Graph(nx.nx_pydot.from_pydot(A))\n\n    '
    msg = 'nx.nx_pydot.from_pydot depends on the pydot package, which has known issues and is not actively maintained.\n\nSee https://github.com/networkx/networkx/issues/5723'
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    if P.get_strict(None):
        multiedges = False
    else:
        multiedges = True
    if P.get_type() == 'graph':
        if multiedges:
            N = nx.MultiGraph()
        else:
            N = nx.Graph()
    elif multiedges:
        N = nx.MultiDiGraph()
    else:
        N = nx.DiGraph()
    name = P.get_name().strip('"')
    if name != '':
        N.name = name
    for p in P.get_node_list():
        n = p.get_name().strip('"')
        if n in ('node', 'graph', 'edge'):
            continue
        N.add_node(n, **p.get_attributes())
    for e in P.get_edge_list():
        u = e.get_source()
        v = e.get_destination()
        attr = e.get_attributes()
        s = []
        d = []
        if isinstance(u, str):
            s.append(u.strip('"'))
        else:
            for unodes in u['nodes']:
                s.append(unodes.strip('"'))
        if isinstance(v, str):
            d.append(v.strip('"'))
        else:
            for vnodes in v['nodes']:
                d.append(vnodes.strip('"'))
        for source_node in s:
            for destination_node in d:
                N.add_edge(source_node, destination_node, **attr)
    pattr = P.get_attributes()
    if pattr:
        N.graph['graph'] = pattr
    try:
        N.graph['node'] = P.get_node_defaults()[0]
    except (IndexError, TypeError):
        pass
    try:
        N.graph['edge'] = P.get_edge_defaults()[0]
    except (IndexError, TypeError):
        pass
    return N

def _check_colon_quotes(s):
    if False:
        for i in range(10):
            print('nop')
    return ':' in s and (s[0] != '"' or s[-1] != '"')

def to_pydot(N):
    if False:
        while True:
            i = 10
    'Returns a pydot graph from a NetworkX graph N.\n\n    Parameters\n    ----------\n    N : NetworkX graph\n      A graph created with NetworkX\n\n    Examples\n    --------\n    >>> K5 = nx.complete_graph(5)\n    >>> P = nx.nx_pydot.to_pydot(K5)\n\n    Notes\n    -----\n\n    '
    import pydot
    msg = 'nx.nx_pydot.to_pydot depends on the pydot package, which has known issues and is not actively maintained.\n\nSee https://github.com/networkx/networkx/issues/5723'
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    if N.is_directed():
        graph_type = 'digraph'
    else:
        graph_type = 'graph'
    strict = nx.number_of_selfloops(N) == 0 and (not N.is_multigraph())
    name = N.name
    graph_defaults = N.graph.get('graph', {})
    if name == '':
        P = pydot.Dot('', graph_type=graph_type, strict=strict, **graph_defaults)
    else:
        P = pydot.Dot(f'"{name}"', graph_type=graph_type, strict=strict, **graph_defaults)
    try:
        P.set_node_defaults(**N.graph['node'])
    except KeyError:
        pass
    try:
        P.set_edge_defaults(**N.graph['edge'])
    except KeyError:
        pass
    for (n, nodedata) in N.nodes(data=True):
        str_nodedata = {str(k): str(v) for (k, v) in nodedata.items()}
        n = str(n)
        raise_error = _check_colon_quotes(n) or any((_check_colon_quotes(k) or _check_colon_quotes(v) for (k, v) in str_nodedata.items()))
        if raise_error:
            raise ValueError(f"""Node names and attributes should not contain ":" unless they are quoted with "".                For example the string 'attribute:data1' should be written as '"attribute:data1"'.                Please refer https://github.com/pydot/pydot/issues/258""")
        p = pydot.Node(n, **str_nodedata)
        P.add_node(p)
    if N.is_multigraph():
        for (u, v, key, edgedata) in N.edges(data=True, keys=True):
            str_edgedata = {str(k): str(v) for (k, v) in edgedata.items() if k != 'key'}
            (u, v) = (str(u), str(v))
            raise_error = _check_colon_quotes(u) or _check_colon_quotes(v) or any((_check_colon_quotes(k) or _check_colon_quotes(val) for (k, val) in str_edgedata.items()))
            if raise_error:
                raise ValueError(f"""Node names and attributes should not contain ":" unless they are quoted with "".                    For example the string 'attribute:data1' should be written as '"attribute:data1"'.                    Please refer https://github.com/pydot/pydot/issues/258""")
            edge = pydot.Edge(u, v, key=str(key), **str_edgedata)
            P.add_edge(edge)
    else:
        for (u, v, edgedata) in N.edges(data=True):
            str_edgedata = {str(k): str(v) for (k, v) in edgedata.items()}
            (u, v) = (str(u), str(v))
            raise_error = _check_colon_quotes(u) or _check_colon_quotes(v) or any((_check_colon_quotes(k) or _check_colon_quotes(val) for (k, val) in str_edgedata.items()))
            if raise_error:
                raise ValueError(f"""Node names and attributes should not contain ":" unless they are quoted with "".                    For example the string 'attribute:data1' should be written as '"attribute:data1"'.                    Please refer https://github.com/pydot/pydot/issues/258""")
            edge = pydot.Edge(u, v, **str_edgedata)
            P.add_edge(edge)
    return P

def graphviz_layout(G, prog='neato', root=None):
    if False:
        i = 10
        return i + 15
    'Create node positions using Pydot and Graphviz.\n\n    Returns a dictionary of positions keyed by node.\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n        The graph for which the layout is computed.\n    prog : string (default: \'neato\')\n        The name of the GraphViz program to use for layout.\n        Options depend on GraphViz version but may include:\n        \'dot\', \'twopi\', \'fdp\', \'sfdp\', \'circo\'\n    root : Node from G or None (default: None)\n        The node of G from which to start some layout algorithms.\n\n    Returns\n    -------\n      Dictionary of (x, y) positions keyed by node.\n\n    Examples\n    --------\n    >>> G = nx.complete_graph(4)\n    >>> pos = nx.nx_pydot.graphviz_layout(G)\n    >>> pos = nx.nx_pydot.graphviz_layout(G, prog="dot")\n\n    Notes\n    -----\n    This is a wrapper for pydot_layout.\n    '
    msg = 'nx.nx_pydot.graphviz_layout depends on the pydot package, which has known issues and is not actively maintained. Consider using nx.nx_agraph.graphviz_layout instead.\n\nSee https://github.com/networkx/networkx/issues/5723'
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    return pydot_layout(G=G, prog=prog, root=root)

def pydot_layout(G, prog='neato', root=None):
    if False:
        while True:
            i = 10
    'Create node positions using :mod:`pydot` and Graphviz.\n\n    Parameters\n    ----------\n    G : Graph\n        NetworkX graph to be laid out.\n    prog : string  (default: \'neato\')\n        Name of the GraphViz command to use for layout.\n        Options depend on GraphViz version but may include:\n        \'dot\', \'twopi\', \'fdp\', \'sfdp\', \'circo\'\n    root : Node from G or None (default: None)\n        The node of G from which to start some layout algorithms.\n\n    Returns\n    -------\n    dict\n        Dictionary of positions keyed by node.\n\n    Examples\n    --------\n    >>> G = nx.complete_graph(4)\n    >>> pos = nx.nx_pydot.pydot_layout(G)\n    >>> pos = nx.nx_pydot.pydot_layout(G, prog="dot")\n\n    Notes\n    -----\n    If you use complex node objects, they may have the same string\n    representation and GraphViz could treat them as the same node.\n    The layout may assign both nodes a single location. See Issue #1568\n    If this occurs in your case, consider relabeling the nodes just\n    for the layout computation using something similar to::\n\n        H = nx.convert_node_labels_to_integers(G, label_attribute=\'node_label\')\n        H_layout = nx.nx_pydot.pydot_layout(G, prog=\'dot\')\n        G_layout = {H.nodes[n][\'node_label\']: p for n, p in H_layout.items()}\n\n    '
    import pydot
    msg = 'nx.nx_pydot.pydot_layout depends on the pydot package, which has known issues and is not actively maintained.\n\nSee https://github.com/networkx/networkx/issues/5723'
    warnings.warn(msg, DeprecationWarning, stacklevel=2)
    P = to_pydot(G)
    if root is not None:
        P.set('root', str(root))
    D_bytes = P.create_dot(prog=prog)
    D = str(D_bytes, encoding=getpreferredencoding())
    if D == '':
        print(f'Graphviz layout with {prog} failed')
        print()
        print('To debug what happened try:')
        print('P = nx.nx_pydot.to_pydot(G)')
        print('P.write_dot("file.dot")')
        print(f'And then run {prog} on file.dot')
        return
    Q_list = pydot.graph_from_dot_data(D)
    assert len(Q_list) == 1
    Q = Q_list[0]
    node_pos = {}
    for n in G.nodes():
        str_n = str(n)
        if _check_colon_quotes(str_n):
            raise ValueError(f"""Node names and node attributes should not contain ":" unless they are quoted with "".                For example the string 'attribute:data1' should be written as '"attribute:data1"'.                Please refer https://github.com/pydot/pydot/issues/258""")
        pydot_node = pydot.Node(str_n).get_name()
        node = Q.get_node(pydot_node)
        if isinstance(node, list):
            node = node[0]
        pos = node.get_pos()[1:-1]
        if pos is not None:
            (xx, yy) = pos.split(',')
            node_pos[n] = (float(xx), float(yy))
    return node_pos