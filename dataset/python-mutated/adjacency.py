import networkx as nx
__all__ = ['adjacency_data', 'adjacency_graph']
_attrs = {'id': 'id', 'key': 'key'}

def adjacency_data(G, attrs=_attrs):
    if False:
        print('Hello World!')
    "Returns data in adjacency format that is suitable for JSON serialization\n    and use in JavaScript documents.\n\n    Parameters\n    ----------\n    G : NetworkX graph\n\n    attrs : dict\n        A dictionary that contains two keys 'id' and 'key'. The corresponding\n        values provide the attribute names for storing NetworkX-internal graph\n        data. The values should be unique. Default value:\n        :samp:`dict(id='id', key='key')`.\n\n        If some user-defined graph data use these attribute names as data keys,\n        they may be silently dropped.\n\n    Returns\n    -------\n    data : dict\n       A dictionary with adjacency formatted data.\n\n    Raises\n    ------\n    NetworkXError\n        If values in attrs are not unique.\n\n    Examples\n    --------\n    >>> from networkx.readwrite import json_graph\n    >>> G = nx.Graph([(1, 2)])\n    >>> data = json_graph.adjacency_data(G)\n\n    To serialize with json\n\n    >>> import json\n    >>> s = json.dumps(data)\n\n    Notes\n    -----\n    Graph, node, and link attributes will be written when using this format\n    but attribute keys must be strings if you want to serialize the resulting\n    data with JSON.\n\n    The default value of attrs will be changed in a future release of NetworkX.\n\n    See Also\n    --------\n    adjacency_graph, node_link_data, tree_data\n    "
    multigraph = G.is_multigraph()
    id_ = attrs['id']
    key = None if not multigraph else attrs['key']
    if id_ == key:
        raise nx.NetworkXError('Attribute names are not unique.')
    data = {}
    data['directed'] = G.is_directed()
    data['multigraph'] = multigraph
    data['graph'] = list(G.graph.items())
    data['nodes'] = []
    data['adjacency'] = []
    for (n, nbrdict) in G.adjacency():
        data['nodes'].append({**G.nodes[n], id_: n})
        adj = []
        if multigraph:
            for (nbr, keys) in nbrdict.items():
                for (k, d) in keys.items():
                    adj.append({**d, id_: nbr, key: k})
        else:
            for (nbr, d) in nbrdict.items():
                adj.append({**d, id_: nbr})
        data['adjacency'].append(adj)
    return data

@nx._dispatch(graphs=None)
def adjacency_graph(data, directed=False, multigraph=True, attrs=_attrs):
    if False:
        i = 10
        return i + 15
    "Returns graph from adjacency data format.\n\n    Parameters\n    ----------\n    data : dict\n        Adjacency list formatted graph data\n\n    directed : bool\n        If True, and direction not specified in data, return a directed graph.\n\n    multigraph : bool\n        If True, and multigraph not specified in data, return a multigraph.\n\n    attrs : dict\n        A dictionary that contains two keys 'id' and 'key'. The corresponding\n        values provide the attribute names for storing NetworkX-internal graph\n        data. The values should be unique. Default value:\n        :samp:`dict(id='id', key='key')`.\n\n    Returns\n    -------\n    G : NetworkX graph\n       A NetworkX graph object\n\n    Examples\n    --------\n    >>> from networkx.readwrite import json_graph\n    >>> G = nx.Graph([(1, 2)])\n    >>> data = json_graph.adjacency_data(G)\n    >>> H = json_graph.adjacency_graph(data)\n\n    Notes\n    -----\n    The default value of attrs will be changed in a future release of NetworkX.\n\n    See Also\n    --------\n    adjacency_graph, node_link_data, tree_data\n    "
    multigraph = data.get('multigraph', multigraph)
    directed = data.get('directed', directed)
    if multigraph:
        graph = nx.MultiGraph()
    else:
        graph = nx.Graph()
    if directed:
        graph = graph.to_directed()
    id_ = attrs['id']
    key = None if not multigraph else attrs['key']
    graph.graph = dict(data.get('graph', []))
    mapping = []
    for d in data['nodes']:
        node_data = d.copy()
        node = node_data.pop(id_)
        mapping.append(node)
        graph.add_node(node)
        graph.nodes[node].update(node_data)
    for (i, d) in enumerate(data['adjacency']):
        source = mapping[i]
        for tdata in d:
            target_data = tdata.copy()
            target = target_data.pop(id_)
            if not multigraph:
                graph.add_edge(source, target)
                graph[source][target].update(target_data)
            else:
                ky = target_data.pop(key, None)
                graph.add_edge(source, target, key=ky)
                graph[source][target][ky].update(target_data)
    return graph