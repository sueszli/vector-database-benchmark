"""Base class for MultiGraph."""
from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import NetworkXError, convert
from networkx.classes.coreviews import MultiAdjacencyView
from networkx.classes.graph import Graph
from networkx.classes.reportviews import MultiDegreeView, MultiEdgeView
__all__ = ['MultiGraph']

class MultiGraph(Graph):
    """
    An undirected graph class that can store multiedges.

    Multiedges are multiple edges between two nodes.  Each edge
    can hold optional data or attributes.

    A MultiGraph holds undirected edges.  Self loops are allowed.

    Nodes can be arbitrary (hashable) Python objects with optional
    key/value attributes. By convention `None` is not used as a node.

    Edges are represented as links between nodes with optional
    key/value attributes, in a MultiGraph each edge has a key to
    distinguish between multiple edges that have the same source and
    destination nodes.

    Parameters
    ----------
    incoming_graph_data : input graph (optional, default: None)
        Data to initialize graph. If None (default) an empty
        graph is created.  The data can be any format that is supported
        by the to_networkx_graph() function, currently including edge list,
        dict of dicts, dict of lists, NetworkX graph, 2D NumPy array,
        SciPy sparse array, or PyGraphviz graph.

    multigraph_input : bool or None (default None)
        Note: Only used when `incoming_graph_data` is a dict.
        If True, `incoming_graph_data` is assumed to be a
        dict-of-dict-of-dict-of-dict structure keyed by
        node to neighbor to edge keys to edge data for multi-edges.
        A NetworkXError is raised if this is not the case.
        If False, :func:`to_networkx_graph` is used to try to determine
        the dict's graph data structure as either a dict-of-dict-of-dict
        keyed by node to neighbor to edge data, or a dict-of-iterable
        keyed by node to neighbors.
        If None, the treatment for True is tried, but if it fails,
        the treatment for False is tried.

    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.

    See Also
    --------
    Graph
    DiGraph
    MultiDiGraph

    Examples
    --------
    Create an empty graph structure (a "null graph") with no nodes and
    no edges.

    >>> G = nx.MultiGraph()

    G can be grown in several ways.

    **Nodes:**

    Add one node at a time:

    >>> G.add_node(1)

    Add the nodes from any container (a list, dict, set or
    even the lines from a file or the nodes from another graph).

    >>> G.add_nodes_from([2, 3])
    >>> G.add_nodes_from(range(100, 110))
    >>> H = nx.path_graph(10)
    >>> G.add_nodes_from(H)

    In addition to strings and integers any hashable Python object
    (except None) can represent a node, e.g. a customized node object,
    or even another Graph.

    >>> G.add_node(H)

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> key = G.add_edge(1, 2)

    a list of edges,

    >>> keys = G.add_edges_from([(1, 2), (1, 3)])

    or a collection of edges,

    >>> keys = G.add_edges_from(H.edges)

    If some edges connect nodes not yet in the graph, the nodes
    are added automatically.  If an edge already exists, an additional
    edge is created and stored using a key to identify the edge.
    By default the key is the lowest unused integer.

    >>> keys = G.add_edges_from([(4, 5, {"route": 28}), (4, 5, {"route": 37})])
    >>> G[4]
    AdjacencyView({3: {0: {}}, 5: {0: {}, 1: {'route': 28}, 2: {'route': 37}}})

    **Attributes:**

    Each graph, node, and edge can hold key/value attribute pairs
    in an associated attribute dictionary (the keys must be hashable).
    By default these are empty, but can be added or changed using
    add_edge, add_node or direct manipulation of the attribute
    dictionaries named graph, node and edge respectively.

    >>> G = nx.MultiGraph(day="Friday")
    >>> G.graph
    {'day': 'Friday'}

    Add node attributes using add_node(), add_nodes_from() or G.nodes

    >>> G.add_node(1, time="5pm")
    >>> G.add_nodes_from([3], time="2pm")
    >>> G.nodes[1]
    {'time': '5pm'}
    >>> G.nodes[1]["room"] = 714
    >>> del G.nodes[1]["room"]  # remove attribute
    >>> list(G.nodes(data=True))
    [(1, {'time': '5pm'}), (3, {'time': '2pm'})]

    Add edge attributes using add_edge(), add_edges_from(), subscript
    notation, or G.edges.

    >>> key = G.add_edge(1, 2, weight=4.7)
    >>> keys = G.add_edges_from([(3, 4), (4, 5)], color="red")
    >>> keys = G.add_edges_from([(1, 2, {"color": "blue"}), (2, 3, {"weight": 8})])
    >>> G[1][2][0]["weight"] = 4.7
    >>> G.edges[1, 2, 0]["weight"] = 4

    Warning: we protect the graph data structure by making `G.edges[1,
    2, 0]` a read-only dict-like structure. However, you can assign to
    attributes in e.g. `G.edges[1, 2, 0]`. Thus, use 2 sets of brackets
    to add/change data attributes: `G.edges[1, 2, 0]['weight'] = 4`.

    **Shortcuts:**

    Many common graph features allow python syntax to speed reporting.

    >>> 1 in G  # check if node in graph
    True
    >>> [n for n in G if n < 3]  # iterate through nodes
    [1, 2]
    >>> len(G)  # number of nodes in graph
    5
    >>> G[1]  # adjacency dict-like view mapping neighbor -> edge key -> edge attributes
    AdjacencyView({2: {0: {'weight': 4}, 1: {'color': 'blue'}}})

    Often the best way to traverse all edges of a graph is via the neighbors.
    The neighbors are reported as an adjacency-dict `G.adj` or `G.adjacency()`.

    >>> for n, nbrsdict in G.adjacency():
    ...     for nbr, keydict in nbrsdict.items():
    ...         for key, eattr in keydict.items():
    ...             if "weight" in eattr:
    ...                 # Do something useful with the edges
    ...                 pass

    But the edges() method is often more convenient:

    >>> for u, v, keys, weight in G.edges(data="weight", keys=True):
    ...     if weight is not None:
    ...         # Do something useful with the edges
    ...         pass

    **Reporting:**

    Simple graph information is obtained using methods and object-attributes.
    Reporting usually provides views instead of containers to reduce memory
    usage. The views update as the graph is updated similarly to dict-views.
    The objects `nodes`, `edges` and `adj` provide access to data attributes
    via lookup (e.g. `nodes[n]`, `edges[u, v, k]`, `adj[u][v]`) and iteration
    (e.g. `nodes.items()`, `nodes.data('color')`,
    `nodes.data('color', default='blue')` and similarly for `edges`)
    Views exist for `nodes`, `edges`, `neighbors()`/`adj` and `degree`.

    For details on these and other miscellaneous methods, see below.

    **Subclasses (Advanced):**

    The MultiGraph class uses a dict-of-dict-of-dict-of-dict data structure.
    The outer dict (node_dict) holds adjacency information keyed by node.
    The next dict (adjlist_dict) represents the adjacency information
    and holds edge_key dicts keyed by neighbor. The edge_key dict holds
    each edge_attr dict keyed by edge key. The inner dict
    (edge_attr_dict) represents the edge data and holds edge attribute
    values keyed by attribute names.

    Each of these four dicts in the dict-of-dict-of-dict-of-dict
    structure can be replaced by a user defined dict-like object.
    In general, the dict-like features should be maintained but
    extra features can be added. To replace one of the dicts create
    a new graph class by changing the class(!) variable holding the
    factory for that dict-like structure. The variable names are
    node_dict_factory, node_attr_dict_factory, adjlist_inner_dict_factory,
    adjlist_outer_dict_factory, edge_key_dict_factory, edge_attr_dict_factory
    and graph_attr_dict_factory.

    node_dict_factory : function, (default: dict)
        Factory function to be used to create the dict containing node
        attributes, keyed by node id.
        It should require no arguments and return a dict-like object

    node_attr_dict_factory: function, (default: dict)
        Factory function to be used to create the node attribute
        dict which holds attribute values keyed by attribute name.
        It should require no arguments and return a dict-like object

    adjlist_outer_dict_factory : function, (default: dict)
        Factory function to be used to create the outer-most dict
        in the data structure that holds adjacency info keyed by node.
        It should require no arguments and return a dict-like object.

    adjlist_inner_dict_factory : function, (default: dict)
        Factory function to be used to create the adjacency list
        dict which holds multiedge key dicts keyed by neighbor.
        It should require no arguments and return a dict-like object.

    edge_key_dict_factory : function, (default: dict)
        Factory function to be used to create the edge key dict
        which holds edge data keyed by edge key.
        It should require no arguments and return a dict-like object.

    edge_attr_dict_factory : function, (default: dict)
        Factory function to be used to create the edge attribute
        dict which holds attribute values keyed by attribute name.
        It should require no arguments and return a dict-like object.

    graph_attr_dict_factory : function, (default: dict)
        Factory function to be used to create the graph attribute
        dict which holds attribute values keyed by attribute name.
        It should require no arguments and return a dict-like object.

    Typically, if your extension doesn't impact the data structure all
    methods will inherited without issue except: `to_directed/to_undirected`.
    By default these methods create a DiGraph/Graph class and you probably
    want them to create your extension of a DiGraph/Graph. To facilitate
    this we define two class variables that you can set in your subclass.

    to_directed_class : callable, (default: DiGraph or MultiDiGraph)
        Class to create a new graph structure in the `to_directed` method.
        If `None`, a NetworkX class (DiGraph or MultiDiGraph) is used.

    to_undirected_class : callable, (default: Graph or MultiGraph)
        Class to create a new graph structure in the `to_undirected` method.
        If `None`, a NetworkX class (Graph or MultiGraph) is used.

    **Subclassing Example**

    Create a low memory graph class that effectively disallows edge
    attributes by using a single attribute dict for all edges.
    This reduces the memory used, but you lose edge attributes.

    >>> class ThinGraph(nx.Graph):
    ...     all_edge_dict = {"weight": 1}
    ...
    ...     def single_edge_dict(self):
    ...         return self.all_edge_dict
    ...
    ...     edge_attr_dict_factory = single_edge_dict
    >>> G = ThinGraph()
    >>> G.add_edge(2, 1)
    >>> G[2][1]
    {'weight': 1}
    >>> G.add_edge(2, 2)
    >>> G[2][1] is G[2][2]
    True
    """
    edge_key_dict_factory = dict

    def to_directed_class(self):
        if False:
            print('Hello World!')
        'Returns the class to use for empty directed copies.\n\n        If you subclass the base classes, use this to designate\n        what directed class to use for `to_directed()` copies.\n        '
        return nx.MultiDiGraph

    def to_undirected_class(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns the class to use for empty undirected copies.\n\n        If you subclass the base classes, use this to designate\n        what directed class to use for `to_directed()` copies.\n        '
        return MultiGraph

    def __init__(self, incoming_graph_data=None, multigraph_input=None, **attr):
        if False:
            while True:
                i = 10
        'Initialize a graph with edges, name, or graph attributes.\n\n        Parameters\n        ----------\n        incoming_graph_data : input graph\n            Data to initialize graph.  If incoming_graph_data=None (default)\n            an empty graph is created.  The data can be an edge list, or any\n            NetworkX graph object.  If the corresponding optional Python\n            packages are installed the data can also be a 2D NumPy array, a\n            SciPy sparse array, or a PyGraphviz graph.\n\n        multigraph_input : bool or None (default None)\n            Note: Only used when `incoming_graph_data` is a dict.\n            If True, `incoming_graph_data` is assumed to be a\n            dict-of-dict-of-dict-of-dict structure keyed by\n            node to neighbor to edge keys to edge data for multi-edges.\n            A NetworkXError is raised if this is not the case.\n            If False, :func:`to_networkx_graph` is used to try to determine\n            the dict\'s graph data structure as either a dict-of-dict-of-dict\n            keyed by node to neighbor to edge data, or a dict-of-iterable\n            keyed by node to neighbors.\n            If None, the treatment for True is tried, but if it fails,\n            the treatment for False is tried.\n\n        attr : keyword arguments, optional (default= no attributes)\n            Attributes to add to graph as key=value pairs.\n\n        See Also\n        --------\n        convert\n\n        Examples\n        --------\n        >>> G = nx.MultiGraph()\n        >>> G = nx.MultiGraph(name="my graph")\n        >>> e = [(1, 2), (1, 2), (2, 3), (3, 4)]  # list of edges\n        >>> G = nx.MultiGraph(e)\n\n        Arbitrary graph attribute pairs (key=value) may be assigned\n\n        >>> G = nx.MultiGraph(e, day="Friday")\n        >>> G.graph\n        {\'day\': \'Friday\'}\n\n        '
        if isinstance(incoming_graph_data, dict) and multigraph_input is not False:
            Graph.__init__(self)
            try:
                convert.from_dict_of_dicts(incoming_graph_data, create_using=self, multigraph_input=True)
                self.graph.update(attr)
            except Exception as err:
                if multigraph_input is True:
                    raise nx.NetworkXError(f'converting multigraph_input raised:\n{type(err)}: {err}')
                Graph.__init__(self, incoming_graph_data, **attr)
        else:
            Graph.__init__(self, incoming_graph_data, **attr)

    @cached_property
    def adj(self):
        if False:
            print('Hello World!')
        'Graph adjacency object holding the neighbors of each node.\n\n        This object is a read-only dict-like structure with node keys\n        and neighbor-dict values.  The neighbor-dict is keyed by neighbor\n        to the edgekey-data-dict.  So `G.adj[3][2][0][\'color\'] = \'blue\'` sets\n        the color of the edge `(3, 2, 0)` to `"blue"`.\n\n        Iterating over G.adj behaves like a dict. Useful idioms include\n        `for nbr, edgesdict in G.adj[n].items():`.\n\n        The neighbor information is also provided by subscripting the graph.\n\n        Examples\n        --------\n        >>> e = [(1, 2), (1, 2), (1, 3), (3, 4)]  # list of edges\n        >>> G = nx.MultiGraph(e)\n        >>> G.edges[1, 2, 0]["weight"] = 3\n        >>> result = set()\n        >>> for edgekey, data in G[1][2].items():\n        ...     result.add(data.get(\'weight\', 1))\n        >>> result\n        {1, 3}\n\n        For directed graphs, `G.adj` holds outgoing (successor) info.\n        '
        return MultiAdjacencyView(self._adj)

    def new_edge_key(self, u, v):
        if False:
            return 10
        'Returns an unused key for edges between nodes `u` and `v`.\n\n        The nodes `u` and `v` do not need to be already in the graph.\n\n        Notes\n        -----\n        In the standard MultiGraph class the new key is the number of existing\n        edges between `u` and `v` (increased if necessary to ensure unused).\n        The first edge will have key 0, then 1, etc. If an edge is removed\n        further new_edge_keys may not be in this order.\n\n        Parameters\n        ----------\n        u, v : nodes\n\n        Returns\n        -------\n        key : int\n        '
        try:
            keydict = self._adj[u][v]
        except KeyError:
            return 0
        key = len(keydict)
        while key in keydict:
            key += 1
        return key

    def add_edge(self, u_for_edge, v_for_edge, key=None, **attr):
        if False:
            return 10
        "Add an edge between u and v.\n\n        The nodes u and v will be automatically added if they are\n        not already in the graph.\n\n        Edge attributes can be specified with keywords or by directly\n        accessing the edge's attribute dictionary. See examples below.\n\n        Parameters\n        ----------\n        u_for_edge, v_for_edge : nodes\n            Nodes can be, for example, strings or numbers.\n            Nodes must be hashable (and not None) Python objects.\n        key : hashable identifier, optional (default=lowest unused integer)\n            Used to distinguish multiedges between a pair of nodes.\n        attr : keyword arguments, optional\n            Edge data (or labels or objects) can be assigned using\n            keyword arguments.\n\n        Returns\n        -------\n        The edge key assigned to the edge.\n\n        See Also\n        --------\n        add_edges_from : add a collection of edges\n\n        Notes\n        -----\n        To replace/update edge data, use the optional key argument\n        to identify a unique edge.  Otherwise a new edge will be created.\n\n        NetworkX algorithms designed for weighted graphs cannot use\n        multigraphs directly because it is not clear how to handle\n        multiedge weights.  Convert to Graph using edge attribute\n        'weight' to enable weighted graph algorithms.\n\n        Default keys are generated using the method `new_edge_key()`.\n        This method can be overridden by subclassing the base class and\n        providing a custom `new_edge_key()` method.\n\n        Examples\n        --------\n        The following each add an additional edge e=(1, 2) to graph G:\n\n        >>> G = nx.MultiGraph()\n        >>> e = (1, 2)\n        >>> ekey = G.add_edge(1, 2)  # explicit two-node form\n        >>> G.add_edge(*e)  # single edge as tuple of two nodes\n        1\n        >>> G.add_edges_from([(1, 2)])  # add edges from iterable container\n        [2]\n\n        Associate data to edges using keywords:\n\n        >>> ekey = G.add_edge(1, 2, weight=3)\n        >>> ekey = G.add_edge(1, 2, key=0, weight=4)  # update data for key=0\n        >>> ekey = G.add_edge(1, 3, weight=7, capacity=15, length=342.7)\n\n        For non-string attribute keys, use subscript notation.\n\n        >>> ekey = G.add_edge(1, 2)\n        >>> G[1][2][0].update({0: 5})\n        >>> G.edges[1, 2, 0].update({0: 5})\n        "
        (u, v) = (u_for_edge, v_for_edge)
        if u not in self._adj:
            if u is None:
                raise ValueError('None cannot be a node')
            self._adj[u] = self.adjlist_inner_dict_factory()
            self._node[u] = self.node_attr_dict_factory()
        if v not in self._adj:
            if v is None:
                raise ValueError('None cannot be a node')
            self._adj[v] = self.adjlist_inner_dict_factory()
            self._node[v] = self.node_attr_dict_factory()
        if key is None:
            key = self.new_edge_key(u, v)
        if v in self._adj[u]:
            keydict = self._adj[u][v]
            datadict = keydict.get(key, self.edge_attr_dict_factory())
            datadict.update(attr)
            keydict[key] = datadict
        else:
            datadict = self.edge_attr_dict_factory()
            datadict.update(attr)
            keydict = self.edge_key_dict_factory()
            keydict[key] = datadict
            self._adj[u][v] = keydict
            self._adj[v][u] = keydict
        return key

    def add_edges_from(self, ebunch_to_add, **attr):
        if False:
            while True:
                i = 10
        'Add all the edges in ebunch_to_add.\n\n        Parameters\n        ----------\n        ebunch_to_add : container of edges\n            Each edge given in the container will be added to the\n            graph. The edges can be:\n\n                - 2-tuples (u, v) or\n                - 3-tuples (u, v, d) for an edge data dict d, or\n                - 3-tuples (u, v, k) for not iterable key k, or\n                - 4-tuples (u, v, k, d) for an edge with data and key k\n\n        attr : keyword arguments, optional\n            Edge data (or labels or objects) can be assigned using\n            keyword arguments.\n\n        Returns\n        -------\n        A list of edge keys assigned to the edges in `ebunch`.\n\n        See Also\n        --------\n        add_edge : add a single edge\n        add_weighted_edges_from : convenient way to add weighted edges\n\n        Notes\n        -----\n        Adding the same edge twice has no effect but any edge data\n        will be updated when each duplicate edge is added.\n\n        Edge attributes specified in an ebunch take precedence over\n        attributes specified via keyword arguments.\n\n        Default keys are generated using the method ``new_edge_key()``.\n        This method can be overridden by subclassing the base class and\n        providing a custom ``new_edge_key()`` method.\n\n        When adding edges from an iterator over the graph you are changing,\n        a `RuntimeError` can be raised with message:\n        `RuntimeError: dictionary changed size during iteration`. This\n        happens when the graph\'s underlying dictionary is modified during\n        iteration. To avoid this error, evaluate the iterator into a separate\n        object, e.g. by using `list(iterator_of_edges)`, and pass this\n        object to `G.add_edges_from`.\n\n        Examples\n        --------\n        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc\n        >>> G.add_edges_from([(0, 1), (1, 2)])  # using a list of edge tuples\n        >>> e = zip(range(0, 3), range(1, 4))\n        >>> G.add_edges_from(e)  # Add the path graph 0-1-2-3\n\n        Associate data to edges\n\n        >>> G.add_edges_from([(1, 2), (2, 3)], weight=3)\n        >>> G.add_edges_from([(3, 4), (1, 4)], label="WN2898")\n\n        Evaluate an iterator over a graph if using it to modify the same graph\n\n        >>> G = nx.MultiGraph([(1, 2), (2, 3), (3, 4)])\n        >>> # Grow graph by one new node, adding edges to all existing nodes.\n        >>> # wrong way - will raise RuntimeError\n        >>> # G.add_edges_from(((5, n) for n in G.nodes))\n        >>> # right way - note that there will be no self-edge for node 5\n        >>> assigned_keys = G.add_edges_from(list((5, n) for n in G.nodes))\n        '
        keylist = []
        for e in ebunch_to_add:
            ne = len(e)
            if ne == 4:
                (u, v, key, dd) = e
            elif ne == 3:
                (u, v, dd) = e
                key = None
            elif ne == 2:
                (u, v) = e
                dd = {}
                key = None
            else:
                msg = f'Edge tuple {e} must be a 2-tuple, 3-tuple or 4-tuple.'
                raise NetworkXError(msg)
            ddd = {}
            ddd.update(attr)
            try:
                ddd.update(dd)
            except (TypeError, ValueError):
                if ne != 3:
                    raise
                key = dd
            key = self.add_edge(u, v, key)
            self[u][v][key].update(ddd)
            keylist.append(key)
        return keylist

    def remove_edge(self, u, v, key=None):
        if False:
            i = 10
            return i + 15
        'Remove an edge between u and v.\n\n        Parameters\n        ----------\n        u, v : nodes\n            Remove an edge between nodes u and v.\n        key : hashable identifier, optional (default=None)\n            Used to distinguish multiple edges between a pair of nodes.\n            If None, remove a single edge between u and v. If there are\n            multiple edges, removes the last edge added in terms of\n            insertion order.\n\n        Raises\n        ------\n        NetworkXError\n            If there is not an edge between u and v, or\n            if there is no edge with the specified key.\n\n        See Also\n        --------\n        remove_edges_from : remove a collection of edges\n\n        Examples\n        --------\n        >>> G = nx.MultiGraph()\n        >>> nx.add_path(G, [0, 1, 2, 3])\n        >>> G.remove_edge(0, 1)\n        >>> e = (1, 2)\n        >>> G.remove_edge(*e)  # unpacks e from an edge tuple\n\n        For multiple edges\n\n        >>> G = nx.MultiGraph()  # or MultiDiGraph, etc\n        >>> G.add_edges_from([(1, 2), (1, 2), (1, 2)])  # key_list returned\n        [0, 1, 2]\n\n        When ``key=None`` (the default), edges are removed in the opposite\n        order that they were added:\n\n        >>> G.remove_edge(1, 2)\n        >>> G.edges(keys=True)\n        MultiEdgeView([(1, 2, 0), (1, 2, 1)])\n        >>> G.remove_edge(2, 1)  # edges are not directed\n        >>> G.edges(keys=True)\n        MultiEdgeView([(1, 2, 0)])\n\n        For edges with keys\n\n        >>> G = nx.MultiGraph()\n        >>> G.add_edge(1, 2, key="first")\n        \'first\'\n        >>> G.add_edge(1, 2, key="second")\n        \'second\'\n        >>> G.remove_edge(1, 2, key="first")\n        >>> G.edges(keys=True)\n        MultiEdgeView([(1, 2, \'second\')])\n\n        '
        try:
            d = self._adj[u][v]
        except KeyError as err:
            raise NetworkXError(f'The edge {u}-{v} is not in the graph.') from err
        if key is None:
            d.popitem()
        else:
            try:
                del d[key]
            except KeyError as err:
                msg = f'The edge {u}-{v} with key {key} is not in the graph.'
                raise NetworkXError(msg) from err
        if len(d) == 0:
            del self._adj[u][v]
            if u != v:
                del self._adj[v][u]

    def remove_edges_from(self, ebunch):
        if False:
            i = 10
            return i + 15
        'Remove all edges specified in ebunch.\n\n        Parameters\n        ----------\n        ebunch: list or container of edge tuples\n            Each edge given in the list or container will be removed\n            from the graph. The edges can be:\n\n                - 2-tuples (u, v) A single edge between u and v is removed.\n                - 3-tuples (u, v, key) The edge identified by key is removed.\n                - 4-tuples (u, v, key, data) where data is ignored.\n\n        See Also\n        --------\n        remove_edge : remove a single edge\n\n        Notes\n        -----\n        Will fail silently if an edge in ebunch is not in the graph.\n\n        Examples\n        --------\n        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc\n        >>> ebunch = [(1, 2), (2, 3)]\n        >>> G.remove_edges_from(ebunch)\n\n        Removing multiple copies of edges\n\n        >>> G = nx.MultiGraph()\n        >>> keys = G.add_edges_from([(1, 2), (1, 2), (1, 2)])\n        >>> G.remove_edges_from([(1, 2), (2, 1)])  # edges aren\'t directed\n        >>> list(G.edges())\n        [(1, 2)]\n        >>> G.remove_edges_from([(1, 2), (1, 2)])  # silently ignore extra copy\n        >>> list(G.edges)  # now empty graph\n        []\n\n        When the edge is a 2-tuple ``(u, v)`` but there are multiple edges between\n        u and v in the graph, the most recent edge (in terms of insertion\n        order) is removed.\n\n        >>> G = nx.MultiGraph()\n        >>> for key in ("x", "y", "a"):\n        ...     k = G.add_edge(0, 1, key=key)\n        >>> G.edges(keys=True)\n        MultiEdgeView([(0, 1, \'x\'), (0, 1, \'y\'), (0, 1, \'a\')])\n        >>> G.remove_edges_from([(0, 1)])\n        >>> G.edges(keys=True)\n        MultiEdgeView([(0, 1, \'x\'), (0, 1, \'y\')])\n\n        '
        for e in ebunch:
            try:
                self.remove_edge(*e[:3])
            except NetworkXError:
                pass

    def has_edge(self, u, v, key=None):
        if False:
            while True:
                i = 10
        'Returns True if the graph has an edge between nodes u and v.\n\n        This is the same as `v in G[u] or key in G[u][v]`\n        without KeyError exceptions.\n\n        Parameters\n        ----------\n        u, v : nodes\n            Nodes can be, for example, strings or numbers.\n\n        key : hashable identifier, optional (default=None)\n            If specified return True only if the edge with\n            key is found.\n\n        Returns\n        -------\n        edge_ind : bool\n            True if edge is in the graph, False otherwise.\n\n        Examples\n        --------\n        Can be called either using two nodes u, v, an edge tuple (u, v),\n        or an edge tuple (u, v, key).\n\n        >>> G = nx.MultiGraph()  # or MultiDiGraph\n        >>> nx.add_path(G, [0, 1, 2, 3])\n        >>> G.has_edge(0, 1)  # using two nodes\n        True\n        >>> e = (0, 1)\n        >>> G.has_edge(*e)  #  e is a 2-tuple (u, v)\n        True\n        >>> G.add_edge(0, 1, key="a")\n        \'a\'\n        >>> G.has_edge(0, 1, key="a")  # specify key\n        True\n        >>> G.has_edge(1, 0, key="a")  # edges aren\'t directed\n        True\n        >>> e = (0, 1, "a")\n        >>> G.has_edge(*e)  # e is a 3-tuple (u, v, \'a\')\n        True\n\n        The following syntax are equivalent:\n\n        >>> G.has_edge(0, 1)\n        True\n        >>> 1 in G[0]  # though this gives :exc:`KeyError` if 0 not in G\n        True\n        >>> 0 in G[1]  # other order; also gives :exc:`KeyError` if 0 not in G\n        True\n\n        '
        try:
            if key is None:
                return v in self._adj[u]
            else:
                return key in self._adj[u][v]
        except KeyError:
            return False

    @cached_property
    def edges(self):
        if False:
            return 10
        'Returns an iterator over the edges.\n\n        edges(self, nbunch=None, data=False, keys=False, default=None)\n\n        The MultiEdgeView provides set-like operations on the edge-tuples\n        as well as edge attribute lookup. When called, it also provides\n        an EdgeDataView object which allows control of access to edge\n        attributes (but does not provide set-like operations).\n        Hence, ``G.edges[u, v, k][\'color\']`` provides the value of the color\n        attribute for the edge from ``u`` to ``v`` with key ``k`` while\n        ``for (u, v, k, c) in G.edges(data=\'color\', keys=True, default="red"):``\n        iterates through all the edges yielding the color attribute with\n        default `\'red\'` if no color attribute exists.\n\n        Edges are returned as tuples with optional data and keys\n        in the order (node, neighbor, key, data). If ``keys=True`` is not\n        provided, the tuples will just be (node, neighbor, data), but\n        multiple tuples with the same node and neighbor will be generated\n        when multiple edges exist between two nodes.\n\n        Parameters\n        ----------\n        nbunch : single node, container, or all nodes (default= all nodes)\n            The view will only report edges from these nodes.\n        data : string or bool, optional (default=False)\n            The edge attribute returned in 3-tuple (u, v, ddict[data]).\n            If True, return edge attribute dict in 3-tuple (u, v, ddict).\n            If False, return 2-tuple (u, v).\n        keys : bool, optional (default=False)\n            If True, return edge keys with each edge, creating (u, v, k)\n            tuples or (u, v, k, d) tuples if data is also requested.\n        default : value, optional (default=None)\n            Value used for edges that don\'t have the requested attribute.\n            Only relevant if data is not True or False.\n\n        Returns\n        -------\n        edges : MultiEdgeView\n            A view of edge attributes, usually it iterates over (u, v)\n            (u, v, k) or (u, v, k, d) tuples of edges, but can also be\n            used for attribute lookup as ``edges[u, v, k][\'foo\']``.\n\n        Notes\n        -----\n        Nodes in nbunch that are not in the graph will be (quietly) ignored.\n        For directed graphs this returns the out-edges.\n\n        Examples\n        --------\n        >>> G = nx.MultiGraph()\n        >>> nx.add_path(G, [0, 1, 2])\n        >>> key = G.add_edge(2, 3, weight=5)\n        >>> key2 = G.add_edge(2, 1, weight=2)  # multi-edge\n        >>> [e for e in G.edges()]\n        [(0, 1), (1, 2), (1, 2), (2, 3)]\n        >>> G.edges.data()  # default data is {} (empty dict)\n        MultiEdgeDataView([(0, 1, {}), (1, 2, {}), (1, 2, {\'weight\': 2}), (2, 3, {\'weight\': 5})])\n        >>> G.edges.data("weight", default=1)\n        MultiEdgeDataView([(0, 1, 1), (1, 2, 1), (1, 2, 2), (2, 3, 5)])\n        >>> G.edges(keys=True)  # default keys are integers\n        MultiEdgeView([(0, 1, 0), (1, 2, 0), (1, 2, 1), (2, 3, 0)])\n        >>> G.edges.data(keys=True)\n        MultiEdgeDataView([(0, 1, 0, {}), (1, 2, 0, {}), (1, 2, 1, {\'weight\': 2}), (2, 3, 0, {\'weight\': 5})])\n        >>> G.edges.data("weight", default=1, keys=True)\n        MultiEdgeDataView([(0, 1, 0, 1), (1, 2, 0, 1), (1, 2, 1, 2), (2, 3, 0, 5)])\n        >>> G.edges([0, 3])  # Note ordering of tuples from listed sources\n        MultiEdgeDataView([(0, 1), (3, 2)])\n        >>> G.edges([0, 3, 2, 1])  # Note ordering of tuples\n        MultiEdgeDataView([(0, 1), (3, 2), (2, 1), (2, 1)])\n        >>> G.edges(0)\n        MultiEdgeDataView([(0, 1)])\n        '
        return MultiEdgeView(self)

    def get_edge_data(self, u, v, key=None, default=None):
        if False:
            i = 10
            return i + 15
        'Returns the attribute dictionary associated with edge (u, v,\n        key).\n\n        If a key is not provided, returns a dictionary mapping edge keys\n        to attribute dictionaries for each edge between u and v.\n\n        This is identical to `G[u][v][key]` except the default is returned\n        instead of an exception is the edge doesn\'t exist.\n\n        Parameters\n        ----------\n        u, v : nodes\n\n        default :  any Python object (default=None)\n            Value to return if the specific edge (u, v, key) is not\n            found, OR if there are no edges between u and v and no key\n            is specified.\n\n        key : hashable identifier, optional (default=None)\n            Return data only for the edge with specified key, as an\n            attribute dictionary (rather than a dictionary mapping keys\n            to attribute dictionaries).\n\n        Returns\n        -------\n        edge_dict : dictionary\n            The edge attribute dictionary, OR a dictionary mapping edge\n            keys to attribute dictionaries for each of those edges if no\n            specific key is provided (even if there\'s only one edge\n            between u and v).\n\n        Examples\n        --------\n        >>> G = nx.MultiGraph()  # or MultiDiGraph\n        >>> key = G.add_edge(0, 1, key="a", weight=7)\n        >>> G[0][1]["a"]  # key=\'a\'\n        {\'weight\': 7}\n        >>> G.edges[0, 1, "a"]  # key=\'a\'\n        {\'weight\': 7}\n\n        Warning: we protect the graph data structure by making\n        `G.edges` and `G[1][2]` read-only dict-like structures.\n        However, you can assign values to attributes in e.g.\n        `G.edges[1, 2, \'a\']` or `G[1][2][\'a\']` using an additional\n        bracket as shown next. You need to specify all edge info\n        to assign to the edge data associated with an edge.\n\n        >>> G[0][1]["a"]["weight"] = 10\n        >>> G.edges[0, 1, "a"]["weight"] = 10\n        >>> G[0][1]["a"]["weight"]\n        10\n        >>> G.edges[1, 0, "a"]["weight"]\n        10\n\n        >>> G = nx.MultiGraph()  # or MultiDiGraph\n        >>> nx.add_path(G, [0, 1, 2, 3])\n        >>> G.edges[0, 1, 0]["weight"] = 5\n        >>> G.get_edge_data(0, 1)\n        {0: {\'weight\': 5}}\n        >>> e = (0, 1)\n        >>> G.get_edge_data(*e)  # tuple form\n        {0: {\'weight\': 5}}\n        >>> G.get_edge_data(3, 0)  # edge not in graph, returns None\n        >>> G.get_edge_data(3, 0, default=0)  # edge not in graph, return default\n        0\n        >>> G.get_edge_data(1, 0, 0)  # specific key gives back\n        {\'weight\': 5}\n        '
        try:
            if key is None:
                return self._adj[u][v]
            else:
                return self._adj[u][v][key]
        except KeyError:
            return default

    @cached_property
    def degree(self):
        if False:
            return 10
        'A DegreeView for the Graph as G.degree or G.degree().\n\n        The node degree is the number of edges adjacent to the node.\n        The weighted node degree is the sum of the edge weights for\n        edges incident to that node.\n\n        This object provides an iterator for (node, degree) as well as\n        lookup for the degree for a single node.\n\n        Parameters\n        ----------\n        nbunch : single node, container, or all nodes (default= all nodes)\n            The view will only report edges incident to these nodes.\n\n        weight : string or None, optional (default=None)\n           The name of an edge attribute that holds the numerical value used\n           as a weight.  If None, then each edge has weight 1.\n           The degree is the sum of the edge weights adjacent to the node.\n\n        Returns\n        -------\n        MultiDegreeView or int\n            If multiple nodes are requested (the default), returns a `MultiDegreeView`\n            mapping nodes to their degree.\n            If a single node is requested, returns the degree of the node as an integer.\n\n        Examples\n        --------\n        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc\n        >>> nx.add_path(G, [0, 1, 2, 3])\n        >>> G.degree(0)  # node 0 with degree 1\n        1\n        >>> list(G.degree([0, 1]))\n        [(0, 1), (1, 2)]\n\n        '
        return MultiDegreeView(self)

    def is_multigraph(self):
        if False:
            print('Hello World!')
        'Returns True if graph is a multigraph, False otherwise.'
        return True

    def is_directed(self):
        if False:
            i = 10
            return i + 15
        'Returns True if graph is directed, False otherwise.'
        return False

    def copy(self, as_view=False):
        if False:
            i = 10
            return i + 15
        'Returns a copy of the graph.\n\n        The copy method by default returns an independent shallow copy\n        of the graph and attributes. That is, if an attribute is a\n        container, that container is shared by the original an the copy.\n        Use Python\'s `copy.deepcopy` for new containers.\n\n        If `as_view` is True then a view is returned instead of a copy.\n\n        Notes\n        -----\n        All copies reproduce the graph structure, but data attributes\n        may be handled in different ways. There are four types of copies\n        of a graph that people might want.\n\n        Deepcopy -- A "deepcopy" copies the graph structure as well as\n        all data attributes and any objects they might contain.\n        The entire graph object is new so that changes in the copy\n        do not affect the original object. (see Python\'s copy.deepcopy)\n\n        Data Reference (Shallow) -- For a shallow copy the graph structure\n        is copied but the edge, node and graph attribute dicts are\n        references to those in the original graph. This saves\n        time and memory but could cause confusion if you change an attribute\n        in one graph and it changes the attribute in the other.\n        NetworkX does not provide this level of shallow copy.\n\n        Independent Shallow -- This copy creates new independent attribute\n        dicts and then does a shallow copy of the attributes. That is, any\n        attributes that are containers are shared between the new graph\n        and the original. This is exactly what `dict.copy()` provides.\n        You can obtain this style copy using:\n\n            >>> G = nx.path_graph(5)\n            >>> H = G.copy()\n            >>> H = G.copy(as_view=False)\n            >>> H = nx.Graph(G)\n            >>> H = G.__class__(G)\n\n        Fresh Data -- For fresh data, the graph structure is copied while\n        new empty data attribute dicts are created. The resulting graph\n        is independent of the original and it has no edge, node or graph\n        attributes. Fresh copies are not enabled. Instead use:\n\n            >>> H = G.__class__()\n            >>> H.add_nodes_from(G)\n            >>> H.add_edges_from(G.edges)\n\n        View -- Inspired by dict-views, graph-views act like read-only\n        versions of the original graph, providing a copy of the original\n        structure without requiring any memory for copying the information.\n\n        See the Python copy module for more information on shallow\n        and deep copies, https://docs.python.org/3/library/copy.html.\n\n        Parameters\n        ----------\n        as_view : bool, optional (default=False)\n            If True, the returned graph-view provides a read-only view\n            of the original graph without actually copying any data.\n\n        Returns\n        -------\n        G : Graph\n            A copy of the graph.\n\n        See Also\n        --------\n        to_directed: return a directed copy of the graph.\n\n        Examples\n        --------\n        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc\n        >>> H = G.copy()\n\n        '
        if as_view is True:
            return nx.graphviews.generic_graph_view(self)
        G = self.__class__()
        G.graph.update(self.graph)
        G.add_nodes_from(((n, d.copy()) for (n, d) in self._node.items()))
        G.add_edges_from(((u, v, key, datadict.copy()) for (u, nbrs) in self._adj.items() for (v, keydict) in nbrs.items() for (key, datadict) in keydict.items()))
        return G

    def to_directed(self, as_view=False):
        if False:
            for i in range(10):
                print('nop')
        'Returns a directed representation of the graph.\n\n        Returns\n        -------\n        G : MultiDiGraph\n            A directed graph with the same name, same nodes, and with\n            each edge (u, v, k, data) replaced by two directed edges\n            (u, v, k, data) and (v, u, k, data).\n\n        Notes\n        -----\n        This returns a "deepcopy" of the edge, node, and\n        graph attributes which attempts to completely copy\n        all of the data and references.\n\n        This is in contrast to the similar D=MultiDiGraph(G) which\n        returns a shallow copy of the data.\n\n        See the Python copy module for more information on shallow\n        and deep copies, https://docs.python.org/3/library/copy.html.\n\n        Warning: If you have subclassed MultiGraph to use dict-like objects\n        in the data structure, those changes do not transfer to the\n        MultiDiGraph created by this method.\n\n        Examples\n        --------\n        >>> G = nx.MultiGraph()\n        >>> G.add_edge(0, 1)\n        0\n        >>> G.add_edge(0, 1)\n        1\n        >>> H = G.to_directed()\n        >>> list(H.edges)\n        [(0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1)]\n\n        If already directed, return a (deep) copy\n\n        >>> G = nx.MultiDiGraph()\n        >>> G.add_edge(0, 1)\n        0\n        >>> H = G.to_directed()\n        >>> list(H.edges)\n        [(0, 1, 0)]\n        '
        graph_class = self.to_directed_class()
        if as_view is True:
            return nx.graphviews.generic_graph_view(self, graph_class)
        G = graph_class()
        G.graph.update(deepcopy(self.graph))
        G.add_nodes_from(((n, deepcopy(d)) for (n, d) in self._node.items()))
        G.add_edges_from(((u, v, key, deepcopy(datadict)) for (u, nbrs) in self.adj.items() for (v, keydict) in nbrs.items() for (key, datadict) in keydict.items()))
        return G

    def to_undirected(self, as_view=False):
        if False:
            i = 10
            return i + 15
        'Returns an undirected copy of the graph.\n\n        Returns\n        -------\n        G : Graph/MultiGraph\n            A deepcopy of the graph.\n\n        See Also\n        --------\n        copy, add_edge, add_edges_from\n\n        Notes\n        -----\n        This returns a "deepcopy" of the edge, node, and\n        graph attributes which attempts to completely copy\n        all of the data and references.\n\n        This is in contrast to the similar `G = nx.MultiGraph(D)`\n        which returns a shallow copy of the data.\n\n        See the Python copy module for more information on shallow\n        and deep copies, https://docs.python.org/3/library/copy.html.\n\n        Warning: If you have subclassed MultiGraph to use dict-like\n        objects in the data structure, those changes do not transfer\n        to the MultiGraph created by this method.\n\n        Examples\n        --------\n        >>> G = nx.MultiGraph([(0, 1), (0, 1), (1, 2)])\n        >>> H = G.to_directed()\n        >>> list(H.edges)\n        [(0, 1, 0), (0, 1, 1), (1, 0, 0), (1, 0, 1), (1, 2, 0), (2, 1, 0)]\n        >>> G2 = H.to_undirected()\n        >>> list(G2.edges)\n        [(0, 1, 0), (0, 1, 1), (1, 2, 0)]\n        '
        graph_class = self.to_undirected_class()
        if as_view is True:
            return nx.graphviews.generic_graph_view(self, graph_class)
        G = graph_class()
        G.graph.update(deepcopy(self.graph))
        G.add_nodes_from(((n, deepcopy(d)) for (n, d) in self._node.items()))
        G.add_edges_from(((u, v, key, deepcopy(datadict)) for (u, nbrs) in self._adj.items() for (v, keydict) in nbrs.items() for (key, datadict) in keydict.items()))
        return G

    def number_of_edges(self, u=None, v=None):
        if False:
            print('Hello World!')
        'Returns the number of edges between two nodes.\n\n        Parameters\n        ----------\n        u, v : nodes, optional (Default=all edges)\n            If u and v are specified, return the number of edges between\n            u and v. Otherwise return the total number of all edges.\n\n        Returns\n        -------\n        nedges : int\n            The number of edges in the graph.  If nodes `u` and `v` are\n            specified return the number of edges between those nodes. If\n            the graph is directed, this only returns the number of edges\n            from `u` to `v`.\n\n        See Also\n        --------\n        size\n\n        Examples\n        --------\n        For undirected multigraphs, this method counts the total number\n        of edges in the graph::\n\n            >>> G = nx.MultiGraph()\n            >>> G.add_edges_from([(0, 1), (0, 1), (1, 2)])\n            [0, 1, 0]\n            >>> G.number_of_edges()\n            3\n\n        If you specify two nodes, this counts the total number of edges\n        joining the two nodes::\n\n            >>> G.number_of_edges(0, 1)\n            2\n\n        For directed multigraphs, this method can count the total number\n        of directed edges from `u` to `v`::\n\n            >>> G = nx.MultiDiGraph()\n            >>> G.add_edges_from([(0, 1), (0, 1), (1, 0)])\n            [0, 1, 0]\n            >>> G.number_of_edges(0, 1)\n            2\n            >>> G.number_of_edges(1, 0)\n            1\n\n        '
        if u is None:
            return self.size()
        try:
            edgedata = self._adj[u][v]
        except KeyError:
            return 0
        return len(edgedata)