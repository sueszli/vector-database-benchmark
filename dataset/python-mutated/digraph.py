"""Base class for directed graphs."""
from copy import deepcopy
from functools import cached_property
import networkx as nx
from networkx import convert
from networkx.classes.coreviews import AdjacencyView
from networkx.classes.graph import Graph
from networkx.classes.reportviews import DiDegreeView, InDegreeView, InEdgeView, OutDegreeView, OutEdgeView
from networkx.exception import NetworkXError
__all__ = ['DiGraph']

class _CachedPropertyResetterAdjAndSucc:
    """Data Descriptor class that syncs and resets cached properties adj and succ

    The cached properties `adj` and `succ` are reset whenever `_adj` or `_succ`
    are set to new objects. In addition, the attributes `_succ` and `_adj`
    are synced so these two names point to the same object.

    This object sits on a class and ensures that any instance of that
    class clears its cached properties "succ" and "adj" whenever the
    underlying instance attributes "_succ" or "_adj" are set to a new object.
    It only affects the set process of the obj._adj and obj._succ attribute.
    All get/del operations act as they normally would.

    For info on Data Descriptors see: https://docs.python.org/3/howto/descriptor.html
    """

    def __set__(self, obj, value):
        if False:
            for i in range(10):
                print('nop')
        od = obj.__dict__
        od['_adj'] = value
        od['_succ'] = value
        if 'adj' in od:
            del od['adj']
        if 'succ' in od:
            del od['succ']

class _CachedPropertyResetterPred:
    """Data Descriptor class for _pred that resets ``pred`` cached_property when needed

    This assumes that the ``cached_property`` ``G.pred`` should be reset whenever
    ``G._pred`` is set to a new value.

    This object sits on a class and ensures that any instance of that
    class clears its cached property "pred" whenever the underlying
    instance attribute "_pred" is set to a new object. It only affects
    the set process of the obj._pred attribute. All get/del operations
    act as they normally would.

    For info on Data Descriptors see: https://docs.python.org/3/howto/descriptor.html
    """

    def __set__(self, obj, value):
        if False:
            i = 10
            return i + 15
        od = obj.__dict__
        od['_pred'] = value
        if 'pred' in od:
            del od['pred']

class DiGraph(Graph):
    """
    Base class for directed graphs.

    A DiGraph stores nodes and edges with optional data, or attributes.

    DiGraphs hold directed edges.  Self loops are allowed but multiple
    (parallel) edges are not.

    Nodes can be arbitrary (hashable) Python objects with optional
    key/value attributes. By convention `None` is not used as a node.

    Edges are represented as links between nodes with optional
    key/value attributes.

    Parameters
    ----------
    incoming_graph_data : input graph (optional, default: None)
        Data to initialize graph. If None (default) an empty
        graph is created.  The data can be any format that is supported
        by the to_networkx_graph() function, currently including edge list,
        dict of dicts, dict of lists, NetworkX graph, 2D NumPy array, SciPy
        sparse matrix, or PyGraphviz graph.

    attr : keyword arguments, optional (default= no attributes)
        Attributes to add to graph as key=value pairs.

    See Also
    --------
    Graph
    MultiGraph
    MultiDiGraph

    Examples
    --------
    Create an empty graph structure (a "null graph") with no nodes and
    no edges.

    >>> G = nx.DiGraph()

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

    >>> G.add_edge(1, 2)

    a list of edges,

    >>> G.add_edges_from([(1, 2), (1, 3)])

    or a collection of edges,

    >>> G.add_edges_from(H.edges)

    If some edges connect nodes not yet in the graph, the nodes
    are added automatically.  There are no errors when adding
    nodes or edges that already exist.

    **Attributes:**

    Each graph, node, and edge can hold key/value attribute pairs
    in an associated attribute dictionary (the keys must be hashable).
    By default these are empty, but can be added or changed using
    add_edge, add_node or direct manipulation of the attribute
    dictionaries named graph, node and edge respectively.

    >>> G = nx.DiGraph(day="Friday")
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

    >>> G.add_edge(1, 2, weight=4.7)
    >>> G.add_edges_from([(3, 4), (4, 5)], color="red")
    >>> G.add_edges_from([(1, 2, {"color": "blue"}), (2, 3, {"weight": 8})])
    >>> G[1][2]["weight"] = 4.7
    >>> G.edges[1, 2]["weight"] = 4

    Warning: we protect the graph data structure by making `G.edges[1, 2]` a
    read-only dict-like structure. However, you can assign to attributes
    in e.g. `G.edges[1, 2]`. Thus, use 2 sets of brackets to add/change
    data attributes: `G.edges[1, 2]['weight'] = 4`
    (For multigraphs: `MG.edges[u, v, key][name] = value`).

    **Shortcuts:**

    Many common graph features allow python syntax to speed reporting.

    >>> 1 in G  # check if node in graph
    True
    >>> [n for n in G if n < 3]  # iterate through nodes
    [1, 2]
    >>> len(G)  # number of nodes in graph
    5

    Often the best way to traverse all edges of a graph is via the neighbors.
    The neighbors are reported as an adjacency-dict `G.adj` or `G.adjacency()`

    >>> for n, nbrsdict in G.adjacency():
    ...     for nbr, eattr in nbrsdict.items():
    ...         if "weight" in eattr:
    ...             # Do something useful with the edges
    ...             pass

    But the edges reporting object is often more convenient:

    >>> for u, v, weight in G.edges(data="weight"):
    ...     if weight is not None:
    ...         # Do something useful with the edges
    ...         pass

    **Reporting:**

    Simple graph information is obtained using object-attributes and methods.
    Reporting usually provides views instead of containers to reduce memory
    usage. The views update as the graph is updated similarly to dict-views.
    The objects `nodes`, `edges` and `adj` provide access to data attributes
    via lookup (e.g. `nodes[n]`, `edges[u, v]`, `adj[u][v]`) and iteration
    (e.g. `nodes.items()`, `nodes.data('color')`,
    `nodes.data('color', default='blue')` and similarly for `edges`)
    Views exist for `nodes`, `edges`, `neighbors()`/`adj` and `degree`.

    For details on these and other miscellaneous methods, see below.

    **Subclasses (Advanced):**

    The Graph class uses a dict-of-dict-of-dict data structure.
    The outer dict (node_dict) holds adjacency information keyed by node.
    The next dict (adjlist_dict) represents the adjacency information and holds
    edge data keyed by neighbor.  The inner dict (edge_attr_dict) represents
    the edge data and holds edge attribute values keyed by attribute names.

    Each of these three dicts can be replaced in a subclass by a user defined
    dict-like object. In general, the dict-like features should be
    maintained but extra features can be added. To replace one of the
    dicts create a new graph class by changing the class(!) variable
    holding the factory for that dict-like structure. The variable names are
    node_dict_factory, node_attr_dict_factory, adjlist_inner_dict_factory,
    adjlist_outer_dict_factory, edge_attr_dict_factory and graph_attr_dict_factory.

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

    adjlist_inner_dict_factory : function, optional (default: dict)
        Factory function to be used to create the adjacency list
        dict which holds edge data keyed by neighbor.
        It should require no arguments and return a dict-like object

    edge_attr_dict_factory : function, optional (default: dict)
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
    _adj = _CachedPropertyResetterAdjAndSucc()
    _succ = _adj
    _pred = _CachedPropertyResetterPred()

    def __init__(self, incoming_graph_data=None, **attr):
        if False:
            while True:
                i = 10
        'Initialize a graph with edges, name, or graph attributes.\n\n        Parameters\n        ----------\n        incoming_graph_data : input graph (optional, default: None)\n            Data to initialize graph.  If None (default) an empty\n            graph is created.  The data can be an edge list, or any\n            NetworkX graph object.  If the corresponding optional Python\n            packages are installed the data can also be a 2D NumPy array, a\n            SciPy sparse array, or a PyGraphviz graph.\n\n        attr : keyword arguments, optional (default= no attributes)\n            Attributes to add to graph as key=value pairs.\n\n        See Also\n        --------\n        convert\n\n        Examples\n        --------\n        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc\n        >>> G = nx.Graph(name="my graph")\n        >>> e = [(1, 2), (2, 3), (3, 4)]  # list of edges\n        >>> G = nx.Graph(e)\n\n        Arbitrary graph attribute pairs (key=value) may be assigned\n\n        >>> G = nx.Graph(e, day="Friday")\n        >>> G.graph\n        {\'day\': \'Friday\'}\n\n        '
        self.graph = self.graph_attr_dict_factory()
        self._node = self.node_dict_factory()
        self._adj = self.adjlist_outer_dict_factory()
        self._pred = self.adjlist_outer_dict_factory()
        if incoming_graph_data is not None:
            convert.to_networkx_graph(incoming_graph_data, create_using=self)
        self.graph.update(attr)

    @cached_property
    def adj(self):
        if False:
            for i in range(10):
                print('nop')
        'Graph adjacency object holding the neighbors of each node.\n\n        This object is a read-only dict-like structure with node keys\n        and neighbor-dict values.  The neighbor-dict is keyed by neighbor\n        to the edge-data-dict.  So `G.adj[3][2][\'color\'] = \'blue\'` sets\n        the color of the edge `(3, 2)` to `"blue"`.\n\n        Iterating over G.adj behaves like a dict. Useful idioms include\n        `for nbr, datadict in G.adj[n].items():`.\n\n        The neighbor information is also provided by subscripting the graph.\n        So `for nbr, foovalue in G[node].data(\'foo\', default=1):` works.\n\n        For directed graphs, `G.adj` holds outgoing (successor) info.\n        '
        return AdjacencyView(self._succ)

    @cached_property
    def succ(self):
        if False:
            while True:
                i = 10
        'Graph adjacency object holding the successors of each node.\n\n        This object is a read-only dict-like structure with node keys\n        and neighbor-dict values.  The neighbor-dict is keyed by neighbor\n        to the edge-data-dict.  So `G.succ[3][2][\'color\'] = \'blue\'` sets\n        the color of the edge `(3, 2)` to `"blue"`.\n\n        Iterating over G.succ behaves like a dict. Useful idioms include\n        `for nbr, datadict in G.succ[n].items():`.  A data-view not provided\n        by dicts also exists: `for nbr, foovalue in G.succ[node].data(\'foo\'):`\n        and a default can be set via a `default` argument to the `data` method.\n\n        The neighbor information is also provided by subscripting the graph.\n        So `for nbr, foovalue in G[node].data(\'foo\', default=1):` works.\n\n        For directed graphs, `G.adj` is identical to `G.succ`.\n        '
        return AdjacencyView(self._succ)

    @cached_property
    def pred(self):
        if False:
            return 10
        'Graph adjacency object holding the predecessors of each node.\n\n        This object is a read-only dict-like structure with node keys\n        and neighbor-dict values.  The neighbor-dict is keyed by neighbor\n        to the edge-data-dict.  So `G.pred[2][3][\'color\'] = \'blue\'` sets\n        the color of the edge `(3, 2)` to `"blue"`.\n\n        Iterating over G.pred behaves like a dict. Useful idioms include\n        `for nbr, datadict in G.pred[n].items():`.  A data-view not provided\n        by dicts also exists: `for nbr, foovalue in G.pred[node].data(\'foo\'):`\n        A default can be set via a `default` argument to the `data` method.\n        '
        return AdjacencyView(self._pred)

    def add_node(self, node_for_adding, **attr):
        if False:
            print('Hello World!')
        'Add a single node `node_for_adding` and update node attributes.\n\n        Parameters\n        ----------\n        node_for_adding : node\n            A node can be any hashable Python object except None.\n        attr : keyword arguments, optional\n            Set or change node attributes using key=value.\n\n        See Also\n        --------\n        add_nodes_from\n\n        Examples\n        --------\n        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc\n        >>> G.add_node(1)\n        >>> G.add_node("Hello")\n        >>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])\n        >>> G.add_node(K3)\n        >>> G.number_of_nodes()\n        3\n\n        Use keywords set/change node attributes:\n\n        >>> G.add_node(1, size=10)\n        >>> G.add_node(3, weight=0.4, UTM=("13S", 382871, 3972649))\n\n        Notes\n        -----\n        A hashable object is one that can be used as a key in a Python\n        dictionary. This includes strings, numbers, tuples of strings\n        and numbers, etc.\n\n        On many platforms hashable items also include mutables such as\n        NetworkX Graphs, though one should be careful that the hash\n        doesn\'t change on mutables.\n        '
        if node_for_adding not in self._succ:
            if node_for_adding is None:
                raise ValueError('None cannot be a node')
            self._succ[node_for_adding] = self.adjlist_inner_dict_factory()
            self._pred[node_for_adding] = self.adjlist_inner_dict_factory()
            attr_dict = self._node[node_for_adding] = self.node_attr_dict_factory()
            attr_dict.update(attr)
        else:
            self._node[node_for_adding].update(attr)

    def add_nodes_from(self, nodes_for_adding, **attr):
        if False:
            return 10
        'Add multiple nodes.\n\n        Parameters\n        ----------\n        nodes_for_adding : iterable container\n            A container of nodes (list, dict, set, etc.).\n            OR\n            A container of (node, attribute dict) tuples.\n            Node attributes are updated using the attribute dict.\n        attr : keyword arguments, optional (default= no attributes)\n            Update attributes for all nodes in nodes.\n            Node attributes specified in nodes as a tuple take\n            precedence over attributes specified via keyword arguments.\n\n        See Also\n        --------\n        add_node\n\n        Notes\n        -----\n        When adding nodes from an iterator over the graph you are changing,\n        a `RuntimeError` can be raised with message:\n        `RuntimeError: dictionary changed size during iteration`. This\n        happens when the graph\'s underlying dictionary is modified during\n        iteration. To avoid this error, evaluate the iterator into a separate\n        object, e.g. by using `list(iterator_of_nodes)`, and pass this\n        object to `G.add_nodes_from`.\n\n        Examples\n        --------\n        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc\n        >>> G.add_nodes_from("Hello")\n        >>> K3 = nx.Graph([(0, 1), (1, 2), (2, 0)])\n        >>> G.add_nodes_from(K3)\n        >>> sorted(G.nodes(), key=str)\n        [0, 1, 2, \'H\', \'e\', \'l\', \'o\']\n\n        Use keywords to update specific node attributes for every node.\n\n        >>> G.add_nodes_from([1, 2], size=10)\n        >>> G.add_nodes_from([3, 4], weight=0.4)\n\n        Use (node, attrdict) tuples to update attributes for specific nodes.\n\n        >>> G.add_nodes_from([(1, dict(size=11)), (2, {"color": "blue"})])\n        >>> G.nodes[1]["size"]\n        11\n        >>> H = nx.Graph()\n        >>> H.add_nodes_from(G.nodes(data=True))\n        >>> H.nodes[1]["size"]\n        11\n\n        Evaluate an iterator over a graph if using it to modify the same graph\n\n        >>> G = nx.DiGraph([(0, 1), (1, 2), (3, 4)])\n        >>> # wrong way - will raise RuntimeError\n        >>> # G.add_nodes_from(n + 1 for n in G.nodes)\n        >>> # correct way\n        >>> G.add_nodes_from(list(n + 1 for n in G.nodes))\n        '
        for n in nodes_for_adding:
            try:
                newnode = n not in self._node
                newdict = attr
            except TypeError:
                (n, ndict) = n
                newnode = n not in self._node
                newdict = attr.copy()
                newdict.update(ndict)
            if newnode:
                if n is None:
                    raise ValueError('None cannot be a node')
                self._succ[n] = self.adjlist_inner_dict_factory()
                self._pred[n] = self.adjlist_inner_dict_factory()
                self._node[n] = self.node_attr_dict_factory()
            self._node[n].update(newdict)

    def remove_node(self, n):
        if False:
            while True:
                i = 10
        'Remove node n.\n\n        Removes the node n and all adjacent edges.\n        Attempting to remove a nonexistent node will raise an exception.\n\n        Parameters\n        ----------\n        n : node\n           A node in the graph\n\n        Raises\n        ------\n        NetworkXError\n           If n is not in the graph.\n\n        See Also\n        --------\n        remove_nodes_from\n\n        Examples\n        --------\n        >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc\n        >>> list(G.edges)\n        [(0, 1), (1, 2)]\n        >>> G.remove_node(1)\n        >>> list(G.edges)\n        []\n\n        '
        try:
            nbrs = self._succ[n]
            del self._node[n]
        except KeyError as err:
            raise NetworkXError(f'The node {n} is not in the digraph.') from err
        for u in nbrs:
            del self._pred[u][n]
        del self._succ[n]
        for u in self._pred[n]:
            del self._succ[u][n]
        del self._pred[n]

    def remove_nodes_from(self, nodes):
        if False:
            i = 10
            return i + 15
        "Remove multiple nodes.\n\n        Parameters\n        ----------\n        nodes : iterable container\n            A container of nodes (list, dict, set, etc.).  If a node\n            in the container is not in the graph it is silently ignored.\n\n        See Also\n        --------\n        remove_node\n\n        Notes\n        -----\n        When removing nodes from an iterator over the graph you are changing,\n        a `RuntimeError` will be raised with message:\n        `RuntimeError: dictionary changed size during iteration`. This\n        happens when the graph's underlying dictionary is modified during\n        iteration. To avoid this error, evaluate the iterator into a separate\n        object, e.g. by using `list(iterator_of_nodes)`, and pass this\n        object to `G.remove_nodes_from`.\n\n        Examples\n        --------\n        >>> G = nx.path_graph(3)  # or DiGraph, MultiGraph, MultiDiGraph, etc\n        >>> e = list(G.nodes)\n        >>> e\n        [0, 1, 2]\n        >>> G.remove_nodes_from(e)\n        >>> list(G.nodes)\n        []\n\n        Evaluate an iterator over a graph if using it to modify the same graph\n\n        >>> G = nx.DiGraph([(0, 1), (1, 2), (3, 4)])\n        >>> # this command will fail, as the graph's dict is modified during iteration\n        >>> # G.remove_nodes_from(n for n in G.nodes if n < 2)\n        >>> # this command will work, since the dictionary underlying graph is not modified\n        >>> G.remove_nodes_from(list(n for n in G.nodes if n < 2))\n        "
        for n in nodes:
            try:
                succs = self._succ[n]
                del self._node[n]
                for u in succs:
                    del self._pred[u][n]
                del self._succ[n]
                for u in self._pred[n]:
                    del self._succ[u][n]
                del self._pred[n]
            except KeyError:
                pass

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        if False:
            while True:
                i = 10
        "Add an edge between u and v.\n\n        The nodes u and v will be automatically added if they are\n        not already in the graph.\n\n        Edge attributes can be specified with keywords or by directly\n        accessing the edge's attribute dictionary. See examples below.\n\n        Parameters\n        ----------\n        u_of_edge, v_of_edge : nodes\n            Nodes can be, for example, strings or numbers.\n            Nodes must be hashable (and not None) Python objects.\n        attr : keyword arguments, optional\n            Edge data (or labels or objects) can be assigned using\n            keyword arguments.\n\n        See Also\n        --------\n        add_edges_from : add a collection of edges\n\n        Notes\n        -----\n        Adding an edge that already exists updates the edge data.\n\n        Many NetworkX algorithms designed for weighted graphs use\n        an edge attribute (by default `weight`) to hold a numerical value.\n\n        Examples\n        --------\n        The following all add the edge e=(1, 2) to graph G:\n\n        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc\n        >>> e = (1, 2)\n        >>> G.add_edge(1, 2)  # explicit two-node form\n        >>> G.add_edge(*e)  # single edge as tuple of two nodes\n        >>> G.add_edges_from([(1, 2)])  # add edges from iterable container\n\n        Associate data to edges using keywords:\n\n        >>> G.add_edge(1, 2, weight=3)\n        >>> G.add_edge(1, 3, weight=7, capacity=15, length=342.7)\n\n        For non-string attribute keys, use subscript notation.\n\n        >>> G.add_edge(1, 2)\n        >>> G[1][2].update({0: 5})\n        >>> G.edges[1, 2].update({0: 5})\n        "
        (u, v) = (u_of_edge, v_of_edge)
        if u not in self._succ:
            if u is None:
                raise ValueError('None cannot be a node')
            self._succ[u] = self.adjlist_inner_dict_factory()
            self._pred[u] = self.adjlist_inner_dict_factory()
            self._node[u] = self.node_attr_dict_factory()
        if v not in self._succ:
            if v is None:
                raise ValueError('None cannot be a node')
            self._succ[v] = self.adjlist_inner_dict_factory()
            self._pred[v] = self.adjlist_inner_dict_factory()
            self._node[v] = self.node_attr_dict_factory()
        datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
        datadict.update(attr)
        self._succ[u][v] = datadict
        self._pred[v][u] = datadict

    def add_edges_from(self, ebunch_to_add, **attr):
        if False:
            for i in range(10):
                print('nop')
        'Add all the edges in ebunch_to_add.\n\n        Parameters\n        ----------\n        ebunch_to_add : container of edges\n            Each edge given in the container will be added to the\n            graph. The edges must be given as 2-tuples (u, v) or\n            3-tuples (u, v, d) where d is a dictionary containing edge data.\n        attr : keyword arguments, optional\n            Edge data (or labels or objects) can be assigned using\n            keyword arguments.\n\n        See Also\n        --------\n        add_edge : add a single edge\n        add_weighted_edges_from : convenient way to add weighted edges\n\n        Notes\n        -----\n        Adding the same edge twice has no effect but any edge data\n        will be updated when each duplicate edge is added.\n\n        Edge attributes specified in an ebunch take precedence over\n        attributes specified via keyword arguments.\n\n        When adding edges from an iterator over the graph you are changing,\n        a `RuntimeError` can be raised with message:\n        `RuntimeError: dictionary changed size during iteration`. This\n        happens when the graph\'s underlying dictionary is modified during\n        iteration. To avoid this error, evaluate the iterator into a separate\n        object, e.g. by using `list(iterator_of_edges)`, and pass this\n        object to `G.add_edges_from`.\n\n        Examples\n        --------\n        >>> G = nx.Graph()  # or DiGraph, MultiGraph, MultiDiGraph, etc\n        >>> G.add_edges_from([(0, 1), (1, 2)])  # using a list of edge tuples\n        >>> e = zip(range(0, 3), range(1, 4))\n        >>> G.add_edges_from(e)  # Add the path graph 0-1-2-3\n\n        Associate data to edges\n\n        >>> G.add_edges_from([(1, 2), (2, 3)], weight=3)\n        >>> G.add_edges_from([(3, 4), (1, 4)], label="WN2898")\n\n        Evaluate an iterator over a graph if using it to modify the same graph\n\n        >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 4)])\n        >>> # Grow graph by one new node, adding edges to all existing nodes.\n        >>> # wrong way - will raise RuntimeError\n        >>> # G.add_edges_from(((5, n) for n in G.nodes))\n        >>> # right way - note that there will be no self-edge for node 5\n        >>> G.add_edges_from(list((5, n) for n in G.nodes))\n        '
        for e in ebunch_to_add:
            ne = len(e)
            if ne == 3:
                (u, v, dd) = e
            elif ne == 2:
                (u, v) = e
                dd = {}
            else:
                raise NetworkXError(f'Edge tuple {e} must be a 2-tuple or 3-tuple.')
            if u not in self._succ:
                if u is None:
                    raise ValueError('None cannot be a node')
                self._succ[u] = self.adjlist_inner_dict_factory()
                self._pred[u] = self.adjlist_inner_dict_factory()
                self._node[u] = self.node_attr_dict_factory()
            if v not in self._succ:
                if v is None:
                    raise ValueError('None cannot be a node')
                self._succ[v] = self.adjlist_inner_dict_factory()
                self._pred[v] = self.adjlist_inner_dict_factory()
                self._node[v] = self.node_attr_dict_factory()
            datadict = self._adj[u].get(v, self.edge_attr_dict_factory())
            datadict.update(attr)
            datadict.update(dd)
            self._succ[u][v] = datadict
            self._pred[v][u] = datadict

    def remove_edge(self, u, v):
        if False:
            i = 10
            return i + 15
        'Remove the edge between u and v.\n\n        Parameters\n        ----------\n        u, v : nodes\n            Remove the edge between nodes u and v.\n\n        Raises\n        ------\n        NetworkXError\n            If there is not an edge between u and v.\n\n        See Also\n        --------\n        remove_edges_from : remove a collection of edges\n\n        Examples\n        --------\n        >>> G = nx.Graph()  # or DiGraph, etc\n        >>> nx.add_path(G, [0, 1, 2, 3])\n        >>> G.remove_edge(0, 1)\n        >>> e = (1, 2)\n        >>> G.remove_edge(*e)  # unpacks e from an edge tuple\n        >>> e = (2, 3, {"weight": 7})  # an edge with attribute data\n        >>> G.remove_edge(*e[:2])  # select first part of edge tuple\n        '
        try:
            del self._succ[u][v]
            del self._pred[v][u]
        except KeyError as err:
            raise NetworkXError(f'The edge {u}-{v} not in graph.') from err

    def remove_edges_from(self, ebunch):
        if False:
            for i in range(10):
                print('nop')
        'Remove all edges specified in ebunch.\n\n        Parameters\n        ----------\n        ebunch: list or container of edge tuples\n            Each edge given in the list or container will be removed\n            from the graph. The edges can be:\n\n                - 2-tuples (u, v) edge between u and v.\n                - 3-tuples (u, v, k) where k is ignored.\n\n        See Also\n        --------\n        remove_edge : remove a single edge\n\n        Notes\n        -----\n        Will fail silently if an edge in ebunch is not in the graph.\n\n        Examples\n        --------\n        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc\n        >>> ebunch = [(1, 2), (2, 3)]\n        >>> G.remove_edges_from(ebunch)\n        '
        for e in ebunch:
            (u, v) = e[:2]
            if u in self._succ and v in self._succ[u]:
                del self._succ[u][v]
                del self._pred[v][u]

    def has_successor(self, u, v):
        if False:
            return 10
        'Returns True if node u has successor v.\n\n        This is true if graph has the edge u->v.\n        '
        return u in self._succ and v in self._succ[u]

    def has_predecessor(self, u, v):
        if False:
            return 10
        'Returns True if node u has predecessor v.\n\n        This is true if graph has the edge u<-v.\n        '
        return u in self._pred and v in self._pred[u]

    def successors(self, n):
        if False:
            i = 10
            return i + 15
        'Returns an iterator over successor nodes of n.\n\n        A successor of n is a node m such that there exists a directed\n        edge from n to m.\n\n        Parameters\n        ----------\n        n : node\n           A node in the graph\n\n        Raises\n        ------\n        NetworkXError\n           If n is not in the graph.\n\n        See Also\n        --------\n        predecessors\n\n        Notes\n        -----\n        neighbors() and successors() are the same.\n        '
        try:
            return iter(self._succ[n])
        except KeyError as err:
            raise NetworkXError(f'The node {n} is not in the digraph.') from err
    neighbors = successors

    def predecessors(self, n):
        if False:
            i = 10
            return i + 15
        'Returns an iterator over predecessor nodes of n.\n\n        A predecessor of n is a node m such that there exists a directed\n        edge from m to n.\n\n        Parameters\n        ----------\n        n : node\n           A node in the graph\n\n        Raises\n        ------\n        NetworkXError\n           If n is not in the graph.\n\n        See Also\n        --------\n        successors\n        '
        try:
            return iter(self._pred[n])
        except KeyError as err:
            raise NetworkXError(f'The node {n} is not in the digraph.') from err

    @cached_property
    def edges(self):
        if False:
            print('Hello World!')
        'An OutEdgeView of the DiGraph as G.edges or G.edges().\n\n        edges(self, nbunch=None, data=False, default=None)\n\n        The OutEdgeView provides set-like operations on the edge-tuples\n        as well as edge attribute lookup. When called, it also provides\n        an EdgeDataView object which allows control of access to edge\n        attributes (but does not provide set-like operations).\n        Hence, `G.edges[u, v][\'color\']` provides the value of the color\n        attribute for edge `(u, v)` while\n        `for (u, v, c) in G.edges.data(\'color\', default=\'red\'):`\n        iterates through all the edges yielding the color attribute\n        with default `\'red\'` if no color attribute exists.\n\n        Parameters\n        ----------\n        nbunch : single node, container, or all nodes (default= all nodes)\n            The view will only report edges from these nodes.\n        data : string or bool, optional (default=False)\n            The edge attribute returned in 3-tuple (u, v, ddict[data]).\n            If True, return edge attribute dict in 3-tuple (u, v, ddict).\n            If False, return 2-tuple (u, v).\n        default : value, optional (default=None)\n            Value used for edges that don\'t have the requested attribute.\n            Only relevant if data is not True or False.\n\n        Returns\n        -------\n        edges : OutEdgeView\n            A view of edge attributes, usually it iterates over (u, v)\n            or (u, v, d) tuples of edges, but can also be used for\n            attribute lookup as `edges[u, v][\'foo\']`.\n\n        See Also\n        --------\n        in_edges, out_edges\n\n        Notes\n        -----\n        Nodes in nbunch that are not in the graph will be (quietly) ignored.\n        For directed graphs this returns the out-edges.\n\n        Examples\n        --------\n        >>> G = nx.DiGraph()  # or MultiDiGraph, etc\n        >>> nx.add_path(G, [0, 1, 2])\n        >>> G.add_edge(2, 3, weight=5)\n        >>> [e for e in G.edges]\n        [(0, 1), (1, 2), (2, 3)]\n        >>> G.edges.data()  # default data is {} (empty dict)\n        OutEdgeDataView([(0, 1, {}), (1, 2, {}), (2, 3, {\'weight\': 5})])\n        >>> G.edges.data("weight", default=1)\n        OutEdgeDataView([(0, 1, 1), (1, 2, 1), (2, 3, 5)])\n        >>> G.edges([0, 2])  # only edges originating from these nodes\n        OutEdgeDataView([(0, 1), (2, 3)])\n        >>> G.edges(0)  # only edges from node 0\n        OutEdgeDataView([(0, 1)])\n\n        '
        return OutEdgeView(self)

    @cached_property
    def out_edges(self):
        if False:
            for i in range(10):
                print('nop')
        return OutEdgeView(self)
    out_edges.__doc__ = edges.__doc__

    @cached_property
    def in_edges(self):
        if False:
            return 10
        "A view of the in edges of the graph as G.in_edges or G.in_edges().\n\n        in_edges(self, nbunch=None, data=False, default=None):\n\n        Parameters\n        ----------\n        nbunch : single node, container, or all nodes (default= all nodes)\n            The view will only report edges incident to these nodes.\n        data : string or bool, optional (default=False)\n            The edge attribute returned in 3-tuple (u, v, ddict[data]).\n            If True, return edge attribute dict in 3-tuple (u, v, ddict).\n            If False, return 2-tuple (u, v).\n        default : value, optional (default=None)\n            Value used for edges that don't have the requested attribute.\n            Only relevant if data is not True or False.\n\n        Returns\n        -------\n        in_edges : InEdgeView or InEdgeDataView\n            A view of edge attributes, usually it iterates over (u, v)\n            or (u, v, d) tuples of edges, but can also be used for\n            attribute lookup as `edges[u, v]['foo']`.\n\n        Examples\n        --------\n        >>> G = nx.DiGraph()\n        >>> G.add_edge(1, 2, color='blue')\n        >>> G.in_edges()\n        InEdgeView([(1, 2)])\n        >>> G.in_edges(nbunch=2)\n        InEdgeDataView([(1, 2)])\n\n        See Also\n        --------\n        edges\n        "
        return InEdgeView(self)

    @cached_property
    def degree(self):
        if False:
            while True:
                i = 10
        'A DegreeView for the Graph as G.degree or G.degree().\n\n        The node degree is the number of edges adjacent to the node.\n        The weighted node degree is the sum of the edge weights for\n        edges incident to that node.\n\n        This object provides an iterator for (node, degree) as well as\n        lookup for the degree for a single node.\n\n        Parameters\n        ----------\n        nbunch : single node, container, or all nodes (default= all nodes)\n            The view will only report edges incident to these nodes.\n\n        weight : string or None, optional (default=None)\n           The name of an edge attribute that holds the numerical value used\n           as a weight.  If None, then each edge has weight 1.\n           The degree is the sum of the edge weights adjacent to the node.\n\n        Returns\n        -------\n        DiDegreeView or int\n            If multiple nodes are requested (the default), returns a `DiDegreeView`\n            mapping nodes to their degree.\n            If a single node is requested, returns the degree of the node as an integer.\n\n        See Also\n        --------\n        in_degree, out_degree\n\n        Examples\n        --------\n        >>> G = nx.DiGraph()  # or MultiDiGraph\n        >>> nx.add_path(G, [0, 1, 2, 3])\n        >>> G.degree(0)  # node 0 with degree 1\n        1\n        >>> list(G.degree([0, 1, 2]))\n        [(0, 1), (1, 2), (2, 2)]\n\n        '
        return DiDegreeView(self)

    @cached_property
    def in_degree(self):
        if False:
            i = 10
            return i + 15
        'An InDegreeView for (node, in_degree) or in_degree for single node.\n\n        The node in_degree is the number of edges pointing to the node.\n        The weighted node degree is the sum of the edge weights for\n        edges incident to that node.\n\n        This object provides an iteration over (node, in_degree) as well as\n        lookup for the degree for a single node.\n\n        Parameters\n        ----------\n        nbunch : single node, container, or all nodes (default= all nodes)\n            The view will only report edges incident to these nodes.\n\n        weight : string or None, optional (default=None)\n           The name of an edge attribute that holds the numerical value used\n           as a weight.  If None, then each edge has weight 1.\n           The degree is the sum of the edge weights adjacent to the node.\n\n        Returns\n        -------\n        If a single node is requested\n        deg : int\n            In-degree of the node\n\n        OR if multiple nodes are requested\n        nd_iter : iterator\n            The iterator returns two-tuples of (node, in-degree).\n\n        See Also\n        --------\n        degree, out_degree\n\n        Examples\n        --------\n        >>> G = nx.DiGraph()\n        >>> nx.add_path(G, [0, 1, 2, 3])\n        >>> G.in_degree(0)  # node 0 with degree 0\n        0\n        >>> list(G.in_degree([0, 1, 2]))\n        [(0, 0), (1, 1), (2, 1)]\n\n        '
        return InDegreeView(self)

    @cached_property
    def out_degree(self):
        if False:
            print('Hello World!')
        'An OutDegreeView for (node, out_degree)\n\n        The node out_degree is the number of edges pointing out of the node.\n        The weighted node degree is the sum of the edge weights for\n        edges incident to that node.\n\n        This object provides an iterator over (node, out_degree) as well as\n        lookup for the degree for a single node.\n\n        Parameters\n        ----------\n        nbunch : single node, container, or all nodes (default= all nodes)\n            The view will only report edges incident to these nodes.\n\n        weight : string or None, optional (default=None)\n           The name of an edge attribute that holds the numerical value used\n           as a weight.  If None, then each edge has weight 1.\n           The degree is the sum of the edge weights adjacent to the node.\n\n        Returns\n        -------\n        If a single node is requested\n        deg : int\n            Out-degree of the node\n\n        OR if multiple nodes are requested\n        nd_iter : iterator\n            The iterator returns two-tuples of (node, out-degree).\n\n        See Also\n        --------\n        degree, in_degree\n\n        Examples\n        --------\n        >>> G = nx.DiGraph()\n        >>> nx.add_path(G, [0, 1, 2, 3])\n        >>> G.out_degree(0)  # node 0 with degree 1\n        1\n        >>> list(G.out_degree([0, 1, 2]))\n        [(0, 1), (1, 1), (2, 1)]\n\n        '
        return OutDegreeView(self)

    def clear(self):
        if False:
            i = 10
            return i + 15
        'Remove all nodes and edges from the graph.\n\n        This also removes the name, and all graph, node, and edge attributes.\n\n        Examples\n        --------\n        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc\n        >>> G.clear()\n        >>> list(G.nodes)\n        []\n        >>> list(G.edges)\n        []\n\n        '
        self._succ.clear()
        self._pred.clear()
        self._node.clear()
        self.graph.clear()

    def clear_edges(self):
        if False:
            print('Hello World!')
        'Remove all edges from the graph without altering nodes.\n\n        Examples\n        --------\n        >>> G = nx.path_graph(4)  # or DiGraph, MultiGraph, MultiDiGraph, etc\n        >>> G.clear_edges()\n        >>> list(G.nodes)\n        [0, 1, 2, 3]\n        >>> list(G.edges)\n        []\n\n        '
        for predecessor_dict in self._pred.values():
            predecessor_dict.clear()
        for successor_dict in self._succ.values():
            successor_dict.clear()

    def is_multigraph(self):
        if False:
            print('Hello World!')
        'Returns True if graph is a multigraph, False otherwise.'
        return False

    def is_directed(self):
        if False:
            i = 10
            return i + 15
        'Returns True if graph is directed, False otherwise.'
        return True

    def to_undirected(self, reciprocal=False, as_view=False):
        if False:
            print('Hello World!')
        'Returns an undirected representation of the digraph.\n\n        Parameters\n        ----------\n        reciprocal : bool (optional)\n          If True only keep edges that appear in both directions\n          in the original digraph.\n        as_view : bool (optional, default=False)\n          If True return an undirected view of the original directed graph.\n\n        Returns\n        -------\n        G : Graph\n            An undirected graph with the same name and nodes and\n            with edge (u, v, data) if either (u, v, data) or (v, u, data)\n            is in the digraph.  If both edges exist in digraph and\n            their edge data is different, only one edge is created\n            with an arbitrary choice of which edge data to use.\n            You must check and correct for this manually if desired.\n\n        See Also\n        --------\n        Graph, copy, add_edge, add_edges_from\n\n        Notes\n        -----\n        If edges in both directions (u, v) and (v, u) exist in the\n        graph, attributes for the new undirected edge will be a combination of\n        the attributes of the directed edges.  The edge data is updated\n        in the (arbitrary) order that the edges are encountered.  For\n        more customized control of the edge attributes use add_edge().\n\n        This returns a "deepcopy" of the edge, node, and\n        graph attributes which attempts to completely copy\n        all of the data and references.\n\n        This is in contrast to the similar G=DiGraph(D) which returns a\n        shallow copy of the data.\n\n        See the Python copy module for more information on shallow\n        and deep copies, https://docs.python.org/3/library/copy.html.\n\n        Warning: If you have subclassed DiGraph to use dict-like objects\n        in the data structure, those changes do not transfer to the\n        Graph created by this method.\n\n        Examples\n        --------\n        >>> G = nx.path_graph(2)  # or MultiGraph, etc\n        >>> H = G.to_directed()\n        >>> list(H.edges)\n        [(0, 1), (1, 0)]\n        >>> G2 = H.to_undirected()\n        >>> list(G2.edges)\n        [(0, 1)]\n        '
        graph_class = self.to_undirected_class()
        if as_view is True:
            return nx.graphviews.generic_graph_view(self, graph_class)
        G = graph_class()
        G.graph.update(deepcopy(self.graph))
        G.add_nodes_from(((n, deepcopy(d)) for (n, d) in self._node.items()))
        if reciprocal is True:
            G.add_edges_from(((u, v, deepcopy(d)) for (u, nbrs) in self._adj.items() for (v, d) in nbrs.items() if v in self._pred[u]))
        else:
            G.add_edges_from(((u, v, deepcopy(d)) for (u, nbrs) in self._adj.items() for (v, d) in nbrs.items()))
        return G

    def reverse(self, copy=True):
        if False:
            i = 10
            return i + 15
        'Returns the reverse of the graph.\n\n        The reverse is a graph with the same nodes and edges\n        but with the directions of the edges reversed.\n\n        Parameters\n        ----------\n        copy : bool optional (default=True)\n            If True, return a new DiGraph holding the reversed edges.\n            If False, the reverse graph is created using a view of\n            the original graph.\n        '
        if copy:
            H = self.__class__()
            H.graph.update(deepcopy(self.graph))
            H.add_nodes_from(((n, deepcopy(d)) for (n, d) in self.nodes.items()))
            H.add_edges_from(((v, u, deepcopy(d)) for (u, v, d) in self.edges(data=True)))
            return H
        return nx.reverse_view(self)