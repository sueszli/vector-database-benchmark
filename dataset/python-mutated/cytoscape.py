import networkx as nx
__all__ = ['cytoscape_data', 'cytoscape_graph']

def cytoscape_data(G, name='name', ident='id'):
    if False:
        return 10
    "Returns data in Cytoscape JSON format (cyjs).\n\n    Parameters\n    ----------\n    G : NetworkX Graph\n        The graph to convert to cytoscape format\n    name : string\n        A string which is mapped to the 'name' node element in cyjs format.\n        Must not have the same value as `ident`.\n    ident : string\n        A string which is mapped to the 'id' node element in cyjs format.\n        Must not have the same value as `name`.\n\n    Returns\n    -------\n    data: dict\n        A dictionary with cyjs formatted data.\n\n    Raises\n    ------\n    NetworkXError\n        If the values for `name` and `ident` are identical.\n\n    See Also\n    --------\n    cytoscape_graph: convert a dictionary in cyjs format to a graph\n\n    References\n    ----------\n    .. [1] Cytoscape user's manual:\n       http://manual.cytoscape.org/en/stable/index.html\n\n    Examples\n    --------\n    >>> G = nx.path_graph(2)\n    >>> nx.cytoscape_data(G)  # doctest: +SKIP\n    {'data': [],\n     'directed': False,\n     'multigraph': False,\n     'elements': {'nodes': [{'data': {'id': '0', 'value': 0, 'name': '0'}},\n       {'data': {'id': '1', 'value': 1, 'name': '1'}}],\n      'edges': [{'data': {'source': 0, 'target': 1}}]}}\n    "
    if name == ident:
        raise nx.NetworkXError('name and ident must be different.')
    jsondata = {'data': list(G.graph.items())}
    jsondata['directed'] = G.is_directed()
    jsondata['multigraph'] = G.is_multigraph()
    jsondata['elements'] = {'nodes': [], 'edges': []}
    nodes = jsondata['elements']['nodes']
    edges = jsondata['elements']['edges']
    for (i, j) in G.nodes.items():
        n = {'data': j.copy()}
        n['data']['id'] = j.get(ident) or str(i)
        n['data']['value'] = i
        n['data']['name'] = j.get(name) or str(i)
        nodes.append(n)
    if G.is_multigraph():
        for e in G.edges(keys=True):
            n = {'data': G.adj[e[0]][e[1]][e[2]].copy()}
            n['data']['source'] = e[0]
            n['data']['target'] = e[1]
            n['data']['key'] = e[2]
            edges.append(n)
    else:
        for e in G.edges():
            n = {'data': G.adj[e[0]][e[1]].copy()}
            n['data']['source'] = e[0]
            n['data']['target'] = e[1]
            edges.append(n)
    return jsondata

@nx._dispatch(graphs=None)
def cytoscape_graph(data, name='name', ident='id'):
    if False:
        return 10
    "\n    Create a NetworkX graph from a dictionary in cytoscape JSON format.\n\n    Parameters\n    ----------\n    data : dict\n        A dictionary of data conforming to cytoscape JSON format.\n    name : string\n        A string which is mapped to the 'name' node element in cyjs format.\n        Must not have the same value as `ident`.\n    ident : string\n        A string which is mapped to the 'id' node element in cyjs format.\n        Must not have the same value as `name`.\n\n    Returns\n    -------\n    graph : a NetworkX graph instance\n        The `graph` can be an instance of `Graph`, `DiGraph`, `MultiGraph`, or\n        `MultiDiGraph` depending on the input data.\n\n    Raises\n    ------\n    NetworkXError\n        If the `name` and `ident` attributes are identical.\n\n    See Also\n    --------\n    cytoscape_data: convert a NetworkX graph to a dict in cyjs format\n\n    References\n    ----------\n    .. [1] Cytoscape user's manual:\n       http://manual.cytoscape.org/en/stable/index.html\n\n    Examples\n    --------\n    >>> data_dict = {\n    ...     'data': [],\n    ...     'directed': False,\n    ...     'multigraph': False,\n    ...     'elements': {'nodes': [{'data': {'id': '0', 'value': 0, 'name': '0'}},\n    ...       {'data': {'id': '1', 'value': 1, 'name': '1'}}],\n    ...      'edges': [{'data': {'source': 0, 'target': 1}}]}\n    ... }\n    >>> G = nx.cytoscape_graph(data_dict)\n    >>> G.name\n    ''\n    >>> G.nodes()\n    NodeView((0, 1))\n    >>> G.nodes(data=True)[0]\n    {'id': '0', 'value': 0, 'name': '0'}\n    >>> G.edges(data=True)\n    EdgeDataView([(0, 1, {'source': 0, 'target': 1})])\n    "
    if name == ident:
        raise nx.NetworkXError('name and ident must be different.')
    multigraph = data.get('multigraph')
    directed = data.get('directed')
    if multigraph:
        graph = nx.MultiGraph()
    else:
        graph = nx.Graph()
    if directed:
        graph = graph.to_directed()
    graph.graph = dict(data.get('data'))
    for d in data['elements']['nodes']:
        node_data = d['data'].copy()
        node = d['data']['value']
        if d['data'].get(name):
            node_data[name] = d['data'].get(name)
        if d['data'].get(ident):
            node_data[ident] = d['data'].get(ident)
        graph.add_node(node)
        graph.nodes[node].update(node_data)
    for d in data['elements']['edges']:
        edge_data = d['data'].copy()
        sour = d['data']['source']
        targ = d['data']['target']
        if multigraph:
            key = d['data'].get('key', 0)
            graph.add_edge(sour, targ, key=key)
            graph.edges[sour, targ, key].update(edge_data)
        else:
            graph.add_edge(sour, targ)
            graph.edges[sour, targ].update(edge_data)
    return graph