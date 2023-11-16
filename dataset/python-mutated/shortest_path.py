from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
from turicreate.data_structures.sgraph import SGraph as _SGraph
from turicreate.toolkits.graph_analytics._model_base import GraphAnalyticsModel as _ModelBase
import copy as _copy

class ShortestPathModel(_ModelBase):
    """
    Model object containing the distance for each vertex in the graph to a
    single source vertex, which is specified during
    :func:`turicreate.shortest_path.create`.

    The model also allows querying for one of the shortest paths from the source
    vertex to any other vertex in the graph.

    Below is a list of queryable fields for this model:

    +----------------+------------------------------------------------------------+
    | Field          | Description                                                |
    +================+============================================================+
    | graph          | A new SGraph with the distance as a vertex property        |
    +----------------+------------------------------------------------------------+
    | distance       | An SFrame with each vertex's distance to the source vertex |
    +----------------+------------------------------------------------------------+
    | weight_field   | The edge field for weight                                  |
    +----------------+------------------------------------------------------------+
    | source_vid     | The source vertex id                                       |
    +----------------+------------------------------------------------------------+
    | max_distance   | Maximum distance between any two vertices                  |
    +----------------+------------------------------------------------------------+
    | training_time  | Total training time of the model                           |
    +----------------+------------------------------------------------------------+

    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.shortest_path.create` to create an instance
    of this model. A detailed list of parameter options and code samples
    are available in the documentation for the create function.

    See Also
    --------
    create
    """

    def __init__(self, model):
        if False:
            print('Hello World!')
        '__init__(self)'
        self.__proxy__ = model
        self._path_query_table = None

    def _result_fields(self):
        if False:
            while True:
                i = 10
        '\n        Return results information\n        Fields should NOT be wrapped by _precomputed_field\n        '
        ret = super(ShortestPathModel, self)._result_fields()
        ret['vertex distance to the source vertex'] = 'SFrame. m.distance'
        return ret

    def _setting_fields(self):
        if False:
            i = 10
            return i + 15
        '\n        Return model fields related to input setting\n        Fields SHOULD be wrapped by _precomputed_field, if necessary\n        '
        ret = super(ShortestPathModel, self)._setting_fields()
        ret['source vertex id'] = 'source_vid'
        ret['edge weight field id'] = 'weight_field'
        ret['maximum distance between vertices'] = 'max_distance'
        return ret

    def _method_fields(self):
        if False:
            while True:
                i = 10
        '\n        Return model fields related to model methods\n        Fields should NOT be wrapped by _precomputed_field\n        '
        return {'get shortest path': 'get_path()  e.g. m.get_path(vid=target_vid)'}

    def get_path(self, vid, highlight=None):
        if False:
            while True:
                i = 10
        '\n        Get the shortest path.\n        Return one of the shortest paths between the source vertex defined\n        in the model and the query vertex.\n        The source vertex is specified by the original call to shortest path.\n        Optionally, plots the path with networkx.\n\n        Parameters\n        ----------\n        vid : string\n            ID of the destination vertex. The source vertex ID is specified\n            when the shortest path result is first computed.\n\n        highlight : list\n            If the path is plotted, identifies the vertices (by vertex ID) that\n            should be highlighted by plotting in a different color.\n\n        Returns\n        -------\n        path : list\n            List of pairs of (vertex_id, distance) in the path.\n\n        Examples\n        --------\n        >>> m.get_path(vid=0)\n        '
        if self._path_query_table is None:
            self._path_query_table = self._generate_path_sframe()
        source_vid = self.source_vid
        path = []
        path_query_table = self._path_query_table
        if not vid in path_query_table['vid']:
            raise ValueError('Destination vertex id ' + str(vid) + ' not found')
        record = path_query_table[path_query_table['vid'] == vid][0]
        dist = record['distance']
        if dist > 100000.0:
            raise ValueError('The distance to {} is too large to show the path.'.format(vid))
        path = [(vid, dist)]
        max_iter = len(path_query_table)
        num_iter = 0
        while record['distance'] != 0 and num_iter < max_iter:
            parent_id = record['parent_row_id']
            assert parent_id < len(path_query_table)
            assert parent_id >= 0
            record = path_query_table[parent_id]
            path.append((record['vid'], record['distance']))
            num_iter += 1
        assert record['vid'] == source_vid
        assert num_iter < max_iter
        path.reverse()
        return path

    def _generate_path_sframe(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Generates an sframe with columns: vid, parent_row_id, and distance.\n        Used for speed up the path query.\n        '
        source_vid = self.source_vid
        weight_field = self.weight_field
        query_table = _copy.copy(self.distance)
        query_table = query_table.add_row_number('row_id')
        g = self.graph.add_vertices(query_table)
        g.vertices['__parent__'] = -1
        weight_field = self.weight_field
        if weight_field == '':
            weight_field = '__unit_weight__'
            g.edges[weight_field] = 1
        traverse_fun = lambda src, edge, dst: _tc.extensions._toolkits.graph.sssp.shortest_path_traverse_function(src, edge, dst, source_vid, weight_field)
        g = g.triple_apply(traverse_fun, ['__parent__'])
        query_table = query_table.join(g.get_vertices()[['__id', '__parent__']], '__id').sort('row_id')
        query_table.rename({'__parent__': 'parent_row_id', '__id': 'vid'}, inplace=True)
        return query_table

    def _get_version(self):
        if False:
            i = 10
            return i + 15
        return 0

    @classmethod
    def _native_name(cls):
        if False:
            return 10
        return 'shortest_path'

    def _get_native_state(self):
        if False:
            i = 10
            return i + 15
        return {'model': self.__proxy__}

    @classmethod
    def _load_version(cls, state, version):
        if False:
            for i in range(10):
                print('nop')
        assert version == 0
        return cls(state['model'])

def create(graph, source_vid, weight_field='', max_distance=1e+30, verbose=True):
    if False:
        print('Hello World!')
    "\n    Compute the single source shortest path distance from the source vertex to\n    all vertices in the graph. Note that because SGraph is directed, shortest\n    paths are also directed. To find undirected shortest paths add edges to the\n    SGraph in both directions. Return a model object with distance each of\n    vertex in the graph.\n\n    Parameters\n    ----------\n    graph : SGraph\n        The graph on which to compute shortest paths.\n\n    source_vid : vertex ID\n        ID of the source vertex.\n\n    weight_field : string, optional\n        The edge field representing the edge weights. If empty, uses unit\n        weights.\n\n    verbose : bool, optional\n        If True, print progress updates.\n\n    Returns\n    -------\n    out : ShortestPathModel\n\n    References\n    ----------\n    - `Wikipedia - ShortestPath <http://en.wikipedia.org/wiki/Shortest_path_problem>`_\n\n    Examples\n    --------\n    If given an :class:`~turicreate.SGraph` ``g``, we can create\n    a :class:`~turicreate.shortest_path.ShortestPathModel` as follows:\n\n    >>> g = turicreate.load_sgraph('http://snap.stanford.edu/data/email-Enron.txt.gz', format='snap')\n    >>> sp = turicreate.shortest_path.create(g, source_vid=1)\n\n    We can obtain the shortest path distance from the source vertex to each\n    vertex in the graph ``g`` as follows:\n\n    >>> sp_sframe = sp['distance']   # SFrame\n\n    We can add the new distance field to the original graph g using:\n\n    >>> g.vertices['distance_to_1'] = sp['graph'].vertices['distance']\n\n    Note that the task above does not require a join because the vertex\n    ordering is preserved through ``create()``.\n\n    To get the actual path from the source vertex to any destination vertex:\n\n    >>> path = sp.get_path(vid=10)\n\n\n    We can obtain an auxiliary graph with additional information corresponding\n    to the shortest path from the source vertex to each vertex in the graph\n    ``g`` as follows:\n\n    >>> sp_graph = sp.get.graph      # SGraph\n\n    See Also\n    --------\n    ShortestPathModel\n    "
    from turicreate._cython.cy_server import QuietProgress
    if not isinstance(graph, _SGraph):
        raise TypeError('graph input must be a SGraph object.')
    opts = {'source_vid': source_vid, 'weight_field': weight_field, 'max_distance': max_distance, 'graph': graph.__proxy__}
    with QuietProgress(verbose):
        params = _tc.extensions._toolkits.graph.sssp.create(opts)
    return ShortestPathModel(params['model'])

def _compute_shortest_path(graph, source_vids, dest_vids, weight_field=''):
    if False:
        return 10
    '\n    Computes shortest paths from any vertex in source_vids to any vertex\n    in dest_vids.  Note that because SGraph is directed, shortest paths are\n    also directed. To find undirected shortest paths add edges to the SGraph in\n    both directions. Returns a list of shortest paths between source_vids\n    and dest_vids.\n\n    Note that this function does not compute all shortest paths between every\n    (source, dest) pair. It computes\n\n    Parameters\n    ----------\n    graph : SGraph\n        The graph on which to compute shortest paths.\n\n    source_vids : vertex ID or list of vertex IDs\n        ID of the source vertices\n\n    dest_vids : vertex ID or list of vertex IDs\n        ID of the destination vertices\n\n    weight_field : str, optional.\n        The edge field representing the edge weights. If empty, uses unit\n        weights.\n\n    Returns\n    -------\n    out :  An SArray of lists of all the same length.\n        Each list describes a path of vertices leading from one source\n        vertex to one destination vertex.\n\n    References\n    ----------\n    - `Wikipedia - ShortestPath <http://en.wikipedia.org/wiki/Shortest_path_problem>`_\n\n    Examples\n    --------\n    If given an :class:`~turicreate.SGraph` ``g``, we can create\n    a :class:`~turicreate.shortest_path.ShortestPathModel` as follows:\n\n    >>> edge_src_ids = [\'src1\', \'src2\',   \'a\', \'b\', \'c\'  ]\n    >>> edge_dst_ids = [   \'a\',    \'b\', \'dst\', \'c\', \'dst\']\n    >>> edges = turicreate.SFrame({\'__src_id\': edge_src_ids, \'__dst_id\': edge_dst_ids})\n    >>> g=tc.SGraph().add_edges(edges)\n    >>> turicreate.shortest_path.compute_shortest_path(g, ["src1","src2"], "dst")\n    [[\'a\',\'dst\']]\n\n    See Also\n    --------\n    ShortestPathModel\n    '
    if type(source_vids) != list:
        source_vids = [source_vids]
    if type(dest_vids) != list:
        dest_vids = [dest_vids]
    return _tc.extensions._toolkits.graph.sssp.all_shortest_paths(graph, source_vids, dest_vids, weight_field)