from itertools import chain, count
import networkx as nx
__all__ = ['node_link_data', 'node_link_graph']
_attrs = {'source': 'source', 'target': 'target', 'name': 'id', 'key': 'key', 'link': 'links'}

def _to_tuple(x):
    if False:
        print('Hello World!')
    'Converts lists to tuples, including nested lists.\n\n    All other non-list inputs are passed through unmodified. This function is\n    intended to be used to convert potentially nested lists from json files\n    into valid nodes.\n\n    Examples\n    --------\n    >>> _to_tuple([1, 2, [3, 4]])\n    (1, 2, (3, 4))\n    '
    if not isinstance(x, tuple | list):
        return x
    return tuple(map(_to_tuple, x))

def node_link_data(G, *, source='source', target='target', name='id', key='key', link='links'):
    if False:
        return 10
    'Returns data in node-link format that is suitable for JSON serialization\n    and use in JavaScript documents.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n    source : string\n        A string that provides the \'source\' attribute name for storing NetworkX-internal graph data.\n    target : string\n        A string that provides the \'target\' attribute name for storing NetworkX-internal graph data.\n    name : string\n        A string that provides the \'name\' attribute name for storing NetworkX-internal graph data.\n    key : string\n        A string that provides the \'key\' attribute name for storing NetworkX-internal graph data.\n    link : string\n        A string that provides the \'link\' attribute name for storing NetworkX-internal graph data.\n\n    Returns\n    -------\n    data : dict\n       A dictionary with node-link formatted data.\n\n    Raises\n    ------\n    NetworkXError\n        If the values of \'source\', \'target\' and \'key\' are not unique.\n\n    Examples\n    --------\n    >>> G = nx.Graph([("A", "B")])\n    >>> data1 = nx.node_link_data(G)\n    >>> data1\n    {\'directed\': False, \'multigraph\': False, \'graph\': {}, \'nodes\': [{\'id\': \'A\'}, {\'id\': \'B\'}], \'links\': [{\'source\': \'A\', \'target\': \'B\'}]}\n\n    To serialize with JSON\n\n    >>> import json\n    >>> s1 = json.dumps(data1)\n    >>> s1\n    \'{"directed": false, "multigraph": false, "graph": {}, "nodes": [{"id": "A"}, {"id": "B"}], "links": [{"source": "A", "target": "B"}]}\'\n\n    A graph can also be serialized by passing `node_link_data` as an encoder function. The two methods are equivalent.\n\n    >>> s1 = json.dumps(G, default=nx.node_link_data)\n    >>> s1\n    \'{"directed": false, "multigraph": false, "graph": {}, "nodes": [{"id": "A"}, {"id": "B"}], "links": [{"source": "A", "target": "B"}]}\'\n\n    The attribute names for storing NetworkX-internal graph data can\n    be specified as keyword options.\n\n    >>> H = nx.gn_graph(2)\n    >>> data2 = nx.node_link_data(H, link="edges", source="from", target="to")\n    >>> data2\n    {\'directed\': True, \'multigraph\': False, \'graph\': {}, \'nodes\': [{\'id\': 0}, {\'id\': 1}], \'edges\': [{\'from\': 1, \'to\': 0}]}\n\n    Notes\n    -----\n    Graph, node, and link attributes are stored in this format.  Note that\n    attribute keys will be converted to strings in order to comply with JSON.\n\n    Attribute \'key\' is only used for multigraphs.\n\n    To use `node_link_data` in conjunction with `node_link_graph`,\n    the keyword names for the attributes must match.\n\n\n    See Also\n    --------\n    node_link_graph, adjacency_data, tree_data\n    '
    multigraph = G.is_multigraph()
    key = None if not multigraph else key
    if len({source, target, key}) < 3:
        raise nx.NetworkXError('Attribute names are not unique.')
    data = {'directed': G.is_directed(), 'multigraph': multigraph, 'graph': G.graph, 'nodes': [{**G.nodes[n], name: n} for n in G]}
    if multigraph:
        data[link] = [{**d, source: u, target: v, key: k} for (u, v, k, d) in G.edges(keys=True, data=True)]
    else:
        data[link] = [{**d, source: u, target: v} for (u, v, d) in G.edges(data=True)]
    return data

@nx._dispatch(graphs=None)
def node_link_graph(data, directed=False, multigraph=True, *, source='source', target='target', name='id', key='key', link='links'):
    if False:
        return 10
    "Returns graph from node-link data format.\n    Useful for de-serialization from JSON.\n\n    Parameters\n    ----------\n    data : dict\n        node-link formatted graph data\n\n    directed : bool\n        If True, and direction not specified in data, return a directed graph.\n\n    multigraph : bool\n        If True, and multigraph not specified in data, return a multigraph.\n\n    source : string\n        A string that provides the 'source' attribute name for storing NetworkX-internal graph data.\n    target : string\n        A string that provides the 'target' attribute name for storing NetworkX-internal graph data.\n    name : string\n        A string that provides the 'name' attribute name for storing NetworkX-internal graph data.\n    key : string\n        A string that provides the 'key' attribute name for storing NetworkX-internal graph data.\n    link : string\n        A string that provides the 'link' attribute name for storing NetworkX-internal graph data.\n\n    Returns\n    -------\n    G : NetworkX graph\n        A NetworkX graph object\n\n    Examples\n    --------\n\n    Create data in node-link format by converting a graph.\n\n    >>> G = nx.Graph([('A', 'B')])\n    >>> data = nx.node_link_data(G)\n    >>> data\n    {'directed': False, 'multigraph': False, 'graph': {}, 'nodes': [{'id': 'A'}, {'id': 'B'}], 'links': [{'source': 'A', 'target': 'B'}]}\n\n    Revert data in node-link format to a graph.\n\n    >>> H = nx.node_link_graph(data)\n    >>> print(H.edges)\n    [('A', 'B')]\n\n    To serialize and deserialize a graph with JSON,\n\n    >>> import json\n    >>> d = json.dumps(node_link_data(G))\n    >>> H = node_link_graph(json.loads(d))\n    >>> print(G.edges, H.edges)\n    [('A', 'B')] [('A', 'B')]\n\n\n    Notes\n    -----\n    Attribute 'key' is only used for multigraphs.\n\n    To use `node_link_data` in conjunction with `node_link_graph`,\n    the keyword names for the attributes must match.\n\n    See Also\n    --------\n    node_link_data, adjacency_data, tree_data\n    "
    multigraph = data.get('multigraph', multigraph)
    directed = data.get('directed', directed)
    if multigraph:
        graph = nx.MultiGraph()
    else:
        graph = nx.Graph()
    if directed:
        graph = graph.to_directed()
    key = None if not multigraph else key
    graph.graph = data.get('graph', {})
    c = count()
    for d in data['nodes']:
        node = _to_tuple(d.get(name, next(c)))
        nodedata = {str(k): v for (k, v) in d.items() if k != name}
        graph.add_node(node, **nodedata)
    for d in data[link]:
        src = tuple(d[source]) if isinstance(d[source], list) else d[source]
        tgt = tuple(d[target]) if isinstance(d[target], list) else d[target]
        if not multigraph:
            edgedata = {str(k): v for (k, v) in d.items() if k != source and k != target}
            graph.add_edge(src, tgt, **edgedata)
        else:
            ky = d.get(key, None)
            edgedata = {str(k): v for (k, v) in d.items() if k != source and k != target and (k != key)}
            graph.add_edge(src, tgt, ky, **edgedata)
    return graph