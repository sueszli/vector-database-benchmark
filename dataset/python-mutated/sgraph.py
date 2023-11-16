"""
This package defines the Turi Create SGraph, Vertex, and Edge objects. The SGraph
is a directed graph, consisting of a set of Vertex objects and Edges that
connect pairs of Vertices. The methods in this module are available from the top
level import of the turicreate package.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from .._connect import main as glconnect
from .sframe import SFrame
from .sarray import SArray
from .gframe import GFrame, VERTEX_GFRAME, EDGE_GFRAME
from .._cython.cy_graph import UnityGraphProxy
from .._cython.context import debug_trace as cython_context
from ..util import _is_non_string_iterable, _make_internal_url
from .._deps import pandas as pd
from .._deps import HAS_PANDAS
import inspect
import copy
import sys
if sys.version_info.major > 2:
    from functools import reduce
_VID_COLUMN = '__id'
_SRC_VID_COLUMN = '__src_id'
_DST_VID_COLUMN = '__dst_id'

class Vertex(object):
    """
    A vertex object, consisting of a vertex ID and a dictionary of vertex
    attributes. The vertex ID can be an integer, string, or float.

    Parameters
    ----------
    vid : int or string or float
        Vertex ID.

    attr : dict, optional
        Vertex attributes. A Dictionary of string keys and values with one of
        the following types: int, float, string, array of floats.

    See Also
    --------
    Edge, SGraph

    Examples
    --------
    >>> from turicreate import SGraph, Vertex, Edge
    >>> g = SGraph()

    >>> verts = [Vertex(0, attr={'breed': 'labrador'}),
                 Vertex(1, attr={'breed': 'labrador'}),
                 Vertex(2, attr={'breed': 'vizsla'})]
    >>> g = g.add_vertices(verts)
    """
    __slots__ = ['vid', 'attr']

    def __init__(self, vid, attr={}, _series=None):
        if False:
            while True:
                i = 10
        '__init__(self, vid, attr={})\n        Construct a new vertex.\n        '
        if not _series is None:
            self.vid = _series[_VID_COLUMN]
            self.attr = _series.to_dict()
            self.attr.pop(_VID_COLUMN)
        else:
            self.vid = vid
            self.attr = attr

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'V(' + str(self.vid) + ', ' + str(self.attr) + ')'

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'V(' + str(self.vid) + ', ' + str(self.attr) + ')'

class Edge(object):
    """
    A directed edge between two Vertex objects. An Edge object consists of a
    source vertex ID, a destination vertex ID, and a dictionary of edge
    attributes.

    Parameters
    ----------
    src_vid : int or string or float
        Source vertex ID.

    dst_vid : int or string or float
        Target vertex ID.

    attr : dict
        Edge attributes. A Dictionary of string keys and values with one of the
        following types: integer, float, string, array of floats.

    See Also
    --------
    Vertex, SGraph

    Examples
    --------
    >>> from turicreate import SGraph, Vertex, Edge

    >>> verts = [Vertex(0, attr={'breed': 'labrador'}),
                 Vertex(1, attr={'breed': 'vizsla'})]
    >>> edges = [Edge(0, 1, attr={'size': 'larger_than'})]

    >>> g = SGraph()
    >>> g = g.add_vertices(verts).add_edges(edges)
    """
    __slots__ = ['src_vid', 'dst_vid', 'attr']

    def __init__(self, src_vid, dst_vid, attr={}, _series=None):
        if False:
            while True:
                i = 10
        '__init__(self, vid, attr={})\n        Construct a new edge.\n        '
        if not _series is None:
            self.src_vid = _series[_SRC_VID_COLUMN]
            self.dst_vid = _series[_DST_VID_COLUMN]
            self.attr = _series.to_dict()
            self.attr.pop(_SRC_VID_COLUMN)
            self.attr.pop(_DST_VID_COLUMN)
        else:
            self.src_vid = src_vid
            self.dst_vid = dst_vid
            self.attr = attr

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'E(' + str(self.src_vid) + ' -> ' + str(self.dst_vid) + ', ' + str(self.attr) + ')'

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'E(' + str(self.src_vid) + ' -> ' + str(self.dst_vid) + ', ' + str(self.attr) + ')'

class SGraph(object):
    """
    A scalable graph data structure. The SGraph data structure allows arbitrary
    dictionary attributes on vertices and edges, provides flexible vertex and
    edge query functions, and seamless transformation to and from
    :class:`~turicreate.SFrame`.

    There are several ways to create an SGraph. The simplest way is to make an
    empty SGraph then add vertices and edges with the :py:func:`add_vertices`
    and :py:func:`add_edges` methods. SGraphs can also be created from vertex
    and edge lists stored in :class:`~turicreate.SFrames`. Columns of these
    SFrames not used as vertex IDs are assumed to be vertex or edge attributes.

    Please see the `User Guide
    <https://apple.github.io/turicreate/docs/userguide/sgraph/sgraph.html>`_
    for a more detailed introduction to creating and working with SGraphs.

    Parameters
    ----------
    vertices : SFrame, optional
        Vertex data. Must include an ID column with the name specified by
        the `vid_field` parameter. Additional columns are treated as vertex
        attributes.

    edges : SFrame, optional
        Edge data. Must include source and destination ID columns as specified
        by `src_field` and `dst_field` parameters. Additional columns are treated
        as edge attributes.

    vid_field : str, optional
        The name of vertex ID column in the `vertices` SFrame.

    src_field : str, optional
        The name of source ID column in the `edges` SFrame.

    dst_field : str, optional
        The name of destination ID column in the `edges` SFrame.

    See Also
    --------
    SFrame

    Notes
    -----
    - SGraphs are *structurally immutable*. In the example below, the
      :func:`~add_vertices` and :func:`~add_edges` commands both return a new
      graph; the old graph gets garbage collected.

    Examples
    --------
    >>> from turicreate import SGraph, Vertex, Edge
    >>> g = SGraph()
    >>> verts = [Vertex(0, attr={'breed': 'labrador'}),
                 Vertex(1, attr={'breed': 'labrador'}),
                 Vertex(2, attr={'breed': 'vizsla'})]
    >>> g = g.add_vertices(verts)
    >>> g = g.add_edges(Edge(1, 2))
    """
    __slots__ = ['__proxy__', '_vertices', '_edges']

    def __init__(self, vertices=None, edges=None, vid_field='__id', src_field='__src_id', dst_field='__dst_id', _proxy=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        __init__(vertices=None, edges=None, vid_field='__id', src_field='__src_id', dst_field='__dst_id')\n\n        By default, construct an empty graph when vertices and edges are None.\n        Otherwise construct an SGraph with given vertices and edges.\n\n        Parameters\n        ----------\n        vertices : SFrame, optional\n            An SFrame containing vertex id columns and optional vertex data\n            columns.\n\n        edges : SFrame, optional\n            An SFrame containing source and target id columns and optional edge\n            data columns.\n\n        vid_field : str, optional\n            The name of vertex id column in the `vertices` SFrame.\n\n        src_field : str, optional\n            The name of source id column in the `edges` SFrame.\n\n        dst_field : str, optional\n            The name of target id column in the `edges` SFrame.\n        "
        if _proxy is None:
            self.__proxy__ = UnityGraphProxy()
            if vertices is not None:
                self.__proxy__ = self.add_vertices(vertices, vid_field).__proxy__
            if edges is not None:
                self.__proxy__ = self.add_edges(edges, src_field, dst_field).__proxy__
        else:
            self.__proxy__ = _proxy
        self._vertices = GFrame(self, VERTEX_GFRAME)
        self._edges = GFrame(self, EDGE_GFRAME)

    def __str__(self):
        if False:
            return 10
        'Returns a readable string representation summarizing the graph.'
        return 'SGraph(%s)' % str(self.summary())

    def __repr__(self):
        if False:
            return 10
        'Returns a readable string representation summarizing the graph.'
        return 'SGraph(%s)\nVertex Fields:%s\nEdge Fields:%s' % (str(self.summary()), str(self.get_vertex_fields()), str(self.get_edge_fields()))

    def __copy__(self):
        if False:
            i = 10
            return i + 15
        return SGraph(_proxy=self.__proxy__)

    def __deepcopy__(self, memo):
        if False:
            return 10
        return self.__copy__()

    def copy(self):
        if False:
            while True:
                i = 10
        '\n        Returns a shallow copy of the SGraph.\n        '
        return self.__copy__()

    @property
    def vertices(self):
        if False:
            print('Hello World!')
        "\n        Special vertex SFrame of the SGraph. Modifying the contents of this\n        SFrame changes the vertex data of the SGraph. To preserve the graph\n        structure, the ``__id`` column of this SFrame is read-only.\n\n        See Also\n        --------\n        Edge\n\n        Examples\n        --------\n        >>> from turicreate import SGraph, Vertex\n        >>> g = SGraph().add_vertices([Vertex('cat', {'fluffy': 1}),\n                                       Vertex('dog', {'fluffy': 1, 'woof': 1}),\n                                       Vertex('hippo', {})])\n\n        Copy the 'woof' vertex attribute into a new 'bark' vertex attribute:\n\n        >>> g.vertices['bark'] = g.vertices['woof']\n\n        Remove the 'woof' attribute:\n\n        >>> del g.vertices['woof']\n\n        Create a new field 'likes_fish':\n\n        >>> g.vertices['likes_fish'] = g.vertices['__id'] == 'cat'\n        +-------+--------+------+------------+\n        |  __id | fluffy | bark | likes_fish |\n        +-------+--------+------+------------+\n        |  dog  |  1.0   | 1.0  |     0      |\n        |  cat  |  1.0   | nan  |     1      |\n        | hippo |  nan   | nan  |     0      |\n        +-------+--------+------+------------+\n\n        Replace missing values with zeros:\n\n        >>> for col in g.vertices.column_names():\n        ...     if col != '__id':\n        ...         g.vertices.fillna(col, 0)\n        +-------+--------+------+------------+\n        |  __id | fluffy | bark | likes_fish |\n        +-------+--------+------+------------+\n        |  dog  |  1.0   | 1.0  |     0      |\n        |  cat  |  1.0   | 0.0  |     1      |\n        | hippo |  0.0   | 0.0  |     0      |\n        +-------+--------+------+------------+\n        "
        return self._vertices

    @property
    def edges(self):
        if False:
            while True:
                i = 10
        "\n        Special edge SFrame of the SGraph. Modifying the contents of this SFrame\n        changes the edge data of the SGraph. To preserve the graph structure,\n        the ``__src_id``, and ``__dst_id`` columns of this SFrame are read-only.\n\n        See Also\n        --------\n        Vertex\n\n\n        Examples\n        --------\n        >>> from turicreate import SGraph, Vertex, Edge\n        >>> g = SGraph()\n        >>> g = g.add_vertices([Vertex(x) for x in ['cat', 'dog', 'fossa']])\n        >>> g = g.add_edges([Edge('cat', 'dog', attr={'relationship': 'dislikes'}),\n                             Edge('dog', 'cat', attr={'relationship': 'likes'}),\n                             Edge('dog', 'fossa', attr={'relationship': 'likes'})])\n        >>> g.edges['size'] = ['smaller than', 'larger than', 'equal to']\n        +----------+----------+--------------+--------------+\n        | __src_id | __dst_id | relationship |     size     |\n        +----------+----------+--------------+--------------+\n        |   cat    |   dog    |   dislikes   | smaller than |\n        |   dog    |   cat    |    likes     | larger than  |\n        |   dog    |  fossa   |    likes     |   equal to   |\n        +----------+----------+--------------+--------------+\n        "
        return self._edges

    def summary(self):
        if False:
            print('Hello World!')
        "\n        Return the number of vertices and edges as a dictionary.\n\n        Returns\n        -------\n        out : dict\n            A dictionary containing the number of vertices and edges.\n\n        See Also\n        --------\n        Vertex, Edge\n\n        Examples\n        --------\n        >>> from turicreate import SGraph, Vertex\n        >>> g = SGraph().add_vertices([Vertex(i) for i in range(10)])\n        >>> n_vertex = g.summary()['num_vertices']\n        10\n        >>> n_edge = g.summary()['num_edges']\n        0\n        "
        ret = self.__proxy__.summary()
        return dict(ret.items())

    def get_vertices(self, ids=[], fields={}, format='sframe'):
        if False:
            print('Hello World!')
        '\n        get_vertices(self, ids=list(), fields={}, format=\'sframe\')\n        Return a collection of vertices and their attributes.\n\n        Parameters\n        ----------\n\n        ids : list [int | float | str] or SArray\n            List of vertex IDs to retrieve. Only vertices in this list will be\n            returned. Also accepts a single vertex id.\n\n        fields : dict | pandas.DataFrame\n            Dictionary specifying equality constraint on field values. For\n            example ``{\'gender\': \'M\'}``, returns only vertices whose \'gender\'\n            field is \'M\'. ``None`` can be used to designate a wild card. For\n            example, {\'relationship\': None} will find all vertices with the\n            field \'relationship\' regardless of the value.\n\n        format : {\'sframe\', \'list\'}\n            Output format. The SFrame output (default) contains a column\n            ``__src_id`` with vertex IDs and a column for each vertex attribute.\n            List output returns a list of Vertex objects.\n\n        Returns\n        -------\n        out : SFrame or list [Vertex]\n            An SFrame or list of Vertex objects.\n\n        See Also\n        --------\n        Vertex, get_edges\n\n        Examples\n        --------\n        Return all vertices in the graph.\n\n        >>> from turicreate import SGraph, Vertex\n        >>> g = SGraph().add_vertices([Vertex(0, attr={\'gender\': \'M\'}),\n                                       Vertex(1, attr={\'gender\': \'F\'}),\n                                       Vertex(2, attr={\'gender\': \'F\'})])\n        >>> g.get_vertices()\n        +------+--------+\n        | __id | gender |\n        +------+--------+\n        |  0   |   M    |\n        |  2   |   F    |\n        |  1   |   F    |\n        +------+--------+\n\n        Return vertices 0 and 2.\n\n        >>> g.get_vertices(ids=[0, 2])\n        +------+--------+\n        | __id | gender |\n        +------+--------+\n        |  0   |   M    |\n        |  2   |   F    |\n        +------+--------+\n\n        Return vertices with the vertex attribute "gender" equal to "M".\n\n        >>> g.get_vertices(fields={\'gender\': \'M\'})\n        +------+--------+\n        | __id | gender |\n        +------+--------+\n        |  0   |   M    |\n        +------+--------+\n        '
        if not _is_non_string_iterable(ids):
            ids = [ids]
        if type(ids) not in (list, SArray):
            raise TypeError('ids must be list or SArray type')
        with cython_context():
            sf = SFrame(_proxy=self.__proxy__.get_vertices(ids, fields))
        if format == 'sframe':
            return sf
        elif format == 'dataframe':
            assert HAS_PANDAS, 'Cannot use dataframe because Pandas is not available or version is too low.'
            if sf.num_rows() == 0:
                return pd.DataFrame()
            else:
                df = sf.head(sf.num_rows()).to_dataframe()
                return df.set_index('__id')
        elif format == 'list':
            return _dataframe_to_vertex_list(sf.to_dataframe())
        else:
            raise ValueError('Invalid format specifier')

    def get_edges(self, src_ids=[], dst_ids=[], fields={}, format='sframe'):
        if False:
            while True:
                i = 10
        '\n        get_edges(self, src_ids=list(), dst_ids=list(), fields={}, format=\'sframe\')\n        Return a collection of edges and their attributes. This function is used\n        to find edges by vertex IDs, filter on edge attributes, or list in-out\n        neighbors of vertex sets.\n\n        Parameters\n        ----------\n        src_ids, dst_ids : list or SArray, optional\n            Parallel arrays of vertex IDs, with each pair corresponding to an\n            edge to fetch. Only edges in this list are returned. ``None`` can be\n            used to designate a wild card. For instance, ``src_ids=[1, 2,\n            None]``, ``dst_ids=[3, None, 5]`` will fetch the edge 1->3, all\n            outgoing edges of 2 and all incoming edges of 5. src_id and dst_id\n            may be left empty, which implies an array of all wild cards.\n\n        fields : dict, optional\n            Dictionary specifying equality constraints on field values. For\n            example, ``{\'relationship\': \'following\'}``, returns only edges whose\n            \'relationship\' field equals \'following\'. ``None`` can be used as a\n            value to designate a wild card. e.g. ``{\'relationship\': None}`` will\n            find all edges with the field \'relationship\' regardless of the\n            value.\n\n        format : {\'sframe\', \'list\'}, optional\n            Output format. The \'sframe\' output (default) contains columns\n            __src_id and __dst_id with edge vertex IDs and a column for each\n            edge attribute. List output returns a list of Edge objects.\n\n        Returns\n        -------\n        out : SFrame | list [Edge]\n            An SFrame or list of edges.\n\n        See Also\n        --------\n        Edge, get_vertices\n\n        Examples\n        --------\n        Return all edges in the graph.\n\n        >>> from turicreate import SGraph, Edge\n        >>> g = SGraph().add_edges([Edge(0, 1, attr={\'rating\': 5}),\n                                    Edge(0, 2, attr={\'rating\': 2}),\n                                    Edge(1, 2)])\n        >>> g.get_edges(src_ids=[None], dst_ids=[None])\n        +----------+----------+--------+\n        | __src_id | __dst_id | rating |\n        +----------+----------+--------+\n        |    0     |    2     |   2    |\n        |    0     |    1     |   5    |\n        |    1     |    2     |  None  |\n        +----------+----------+--------+\n\n        Return edges with the attribute "rating" of 5.\n\n        >>> g.get_edges(fields={\'rating\': 5})\n        +----------+----------+--------+\n        | __src_id | __dst_id | rating |\n        +----------+----------+--------+\n        |    0     |    1     |   5    |\n        +----------+----------+--------+\n\n        Return edges 0 --> 1 and 1 --> 2 (if present in the graph).\n\n        >>> g.get_edges(src_ids=[0, 1], dst_ids=[1, 2])\n        +----------+----------+--------+\n        | __src_id | __dst_id | rating |\n        +----------+----------+--------+\n        |    0     |    1     |   5    |\n        |    1     |    2     |  None  |\n        +----------+----------+--------+\n        '
        if not _is_non_string_iterable(src_ids):
            src_ids = [src_ids]
        if not _is_non_string_iterable(dst_ids):
            dst_ids = [dst_ids]
        if type(src_ids) not in (list, SArray):
            raise TypeError('src_ids must be list or SArray type')
        if type(dst_ids) not in (list, SArray):
            raise TypeError('dst_ids must be list or SArray type')
        if len(src_ids) == 0 and len(dst_ids) > 0:
            src_ids = [None] * len(dst_ids)
        if len(dst_ids) == 0 and len(src_ids) > 0:
            dst_ids = [None] * len(src_ids)
        with cython_context():
            sf = SFrame(_proxy=self.__proxy__.get_edges(src_ids, dst_ids, fields))
        if format == 'sframe':
            return sf
        if format == 'dataframe':
            assert HAS_PANDAS, 'Cannot use dataframe because Pandas is not available or version is too low.'
            if sf.num_rows() == 0:
                return pd.DataFrame()
            else:
                return sf.head(sf.num_rows()).to_dataframe()
        elif format == 'list':
            return _dataframe_to_edge_list(sf.to_dataframe())
        else:
            raise ValueError('Invalid format specifier')

    def add_vertices(self, vertices, vid_field=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Add vertices to the SGraph. Vertices should be input as a list of\n        :class:`~turicreate.Vertex` objects, an :class:`~turicreate.SFrame`, or a\n        pandas DataFrame. If vertices are specified by SFrame or DataFrame,\n        ``vid_field`` specifies which column contains the vertex ID. Remaining\n        columns are assumed to hold additional vertex attributes. If these\n        attributes are not already present in the graph's vertex data, they are\n        added, with existing vertices acquiring the value ``None``.\n\n        Parameters\n        ----------\n        vertices : Vertex | list [Vertex] | pandas.DataFrame | SFrame\n            Vertex data. If the vertices are in an SFrame or DataFrame, then\n            ``vid_field`` specifies the column containing the vertex IDs.\n            Additional columns are treated as vertex attributes.\n\n        vid_field : string, optional\n            Column in the DataFrame or SFrame to use as vertex ID. Required if\n            vertices is an SFrame. If ``vertices`` is a DataFrame and\n            ``vid_field`` is not specified, the row index is used as vertex ID.\n\n        Returns\n        -------\n        out : SGraph\n            A new SGraph with vertices added.\n\n        See Also\n        --------\n        Vertex, SFrame, add_edges\n\n        Notes\n        -----\n        - If vertices are added with indices that already exist in the graph,\n          they are overwritten completely. All attributes for these vertices\n          will conform to the specification in this method.\n\n        Examples\n        --------\n        >>> from turicreate import SGraph, Vertex, SFrame\n        >>> g = SGraph()\n\n        Add a single vertex.\n\n        >>> g = g.add_vertices(Vertex(0, attr={'breed': 'labrador'}))\n\n        Add a list of vertices.\n\n        >>> verts = [Vertex(0, attr={'breed': 'labrador'}),\n                     Vertex(1, attr={'breed': 'labrador'}),\n                     Vertex(2, attr={'breed': 'vizsla'})]\n        >>> g = g.add_vertices(verts)\n\n        Add vertices from an SFrame.\n\n        >>> sf_vert = SFrame({'id': [0, 1, 2], 'breed':['lab', 'lab', 'vizsla']})\n        >>> g = g.add_vertices(sf_vert, vid_field='id')\n        "
        sf = _vertex_data_to_sframe(vertices, vid_field)
        with cython_context():
            proxy = self.__proxy__.add_vertices(sf.__proxy__, _VID_COLUMN)
            return SGraph(_proxy=proxy)

    def add_edges(self, edges, src_field=None, dst_field=None):
        if False:
            i = 10
            return i + 15
        "\n        Add edges to the SGraph. Edges should be input as a list of\n        :class:`~turicreate.Edge` objects, an :class:`~turicreate.SFrame`, or a\n        Pandas DataFrame. If the new edges are in an SFrame or DataFrame, then\n        ``src_field`` and ``dst_field`` are required to specify the columns that\n        contain the source and destination vertex IDs; additional columns are\n        treated as edge attributes. If these attributes are not already present\n        in the graph's edge data, they are added, with existing edges acquiring\n        the value ``None``.\n\n        Parameters\n        ----------\n        edges : Edge | list [Edge] | pandas.DataFrame | SFrame\n            Edge data. If the edges are in an SFrame or DataFrame, then\n            ``src_field`` and ``dst_field`` are required to specify the columns\n            that contain the source and destination vertex IDs. Additional\n            columns are treated as edge attributes.\n\n        src_field : string, optional\n            Column in the SFrame or DataFrame to use as source vertex IDs. Not\n            required if ``edges`` is a list.\n\n        dst_field : string, optional\n            Column in the SFrame or Pandas DataFrame to use as destination\n            vertex IDs. Not required if ``edges`` is a list.\n\n        Returns\n        -------\n        out : SGraph\n            A new SGraph with `edges` added.\n\n        See Also\n        --------\n        Edge, SFrame, add_vertices\n\n        Notes\n        -----\n        - If an edge is added whose source and destination IDs match edges that\n          already exist in the graph, a new edge is added to the graph. This\n          contrasts with :py:func:`add_vertices`, which overwrites existing\n          vertices.\n\n        Examples\n        --------\n        >>> from turicreate import SGraph, Vertex, Edge, SFrame\n        >>> g = SGraph()\n        >>> verts = [Vertex(0, attr={'breed': 'labrador'}),\n                     Vertex(1, attr={'breed': 'labrador'}),\n                     Vertex(2, attr={'breed': 'vizsla'})]\n        >>> g = g.add_vertices(verts)\n\n        Add a single edge.\n\n        >>> g = g.add_edges(Edge(1, 2))\n\n        Add a list of edges.\n\n        >>> g = g.add_edges([Edge(0, 2), Edge(1, 2)])\n\n        Add edges from an SFrame.\n\n        >>> sf_edge = SFrame({'source': [0, 1], 'dest': [2, 2]})\n        >>> g = g.add_edges(sf_edge, src_field='source', dst_field='dest')\n        "
        sf = _edge_data_to_sframe(edges, src_field, dst_field)
        with cython_context():
            proxy = self.__proxy__.add_edges(sf.__proxy__, _SRC_VID_COLUMN, _DST_VID_COLUMN)
            return SGraph(_proxy=proxy)

    def get_fields(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a list of vertex and edge attribute fields in the SGraph. If a\n        field is common to both vertex and edge attributes, it will show up\n        twice in the returned list.\n\n        Returns\n        -------\n        out : list\n            Names of fields contained in the vertex or edge data.\n\n        See Also\n        --------\n        get_vertex_fields, get_edge_fields\n\n        Examples\n        --------\n        >>> from turicreate import SGraph, Vertex, Edge\n        >>> g = SGraph()\n        >>> verts = [Vertex(0, attr={'name': 'alex'}),\n                     Vertex(1, attr={'name': 'barbara'})]\n        >>> g = g.add_vertices(verts)\n        >>> g = g.add_edges(Edge(0, 1, attr={'frequency': 6}))\n        >>> fields = g.get_fields()\n        ['__id', 'name', '__src_id', '__dst_id', 'frequency']\n        "
        return self.get_vertex_fields() + self.get_edge_fields()

    def get_vertex_fields(self):
        if False:
            i = 10
            return i + 15
        "\n        Return a list of vertex attribute fields in the SGraph.\n\n        Returns\n        -------\n        out : list\n            Names of fields contained in the vertex data.\n\n        See Also\n        --------\n        get_fields, get_edge_fields\n\n        Examples\n        --------\n        >>> from turicreate import SGraph, Vertex, Edge\n        >>> g = SGraph()\n        >>> verts = [Vertex(0, attr={'name': 'alex'}),\n                     Vertex(1, attr={'name': 'barbara'})]\n        >>> g = g.add_vertices(verts)\n        >>> g = g.add_edges(Edge(0, 1, attr={'frequency': 6}))\n        >>> fields = g.get_vertex_fields()\n        ['__id', 'name']\n        "
        with cython_context():
            return self.__proxy__.get_vertex_fields()

    def get_edge_fields(self):
        if False:
            print('Hello World!')
        "\n        Return a list of edge attribute fields in the graph.\n\n        Returns\n        -------\n        out : list\n            Names of fields contained in the vertex data.\n\n        See Also\n        --------\n        get_fields, get_vertex_fields\n\n        Examples\n        --------\n        >>> from turicreate import SGraph, Vertex, Edge\n        >>> g = SGraph()\n        >>> verts = [Vertex(0, attr={'name': 'alex'}),\n                     Vertex(1, attr={'name': 'barbara'})]\n        >>> g = g.add_vertices(verts)\n        >>> g = g.add_edges(Edge(0, 1, attr={'frequency': 6}))\n        >>> fields = g.get_vertex_fields()\n        ['__src_id', '__dst_id', 'frequency']\n        "
        with cython_context():
            return self.__proxy__.get_edge_fields()

    def select_fields(self, fields):
        if False:
            i = 10
            return i + 15
        "\n        Return a new SGraph with only the selected fields. Other fields are\n        discarded, while fields that do not exist in the SGraph are ignored.\n\n        Parameters\n        ----------\n        fields : string | list [string]\n            A single field name or a list of field names to select.\n\n        Returns\n        -------\n        out : SGraph\n            A new graph whose vertex and edge data are projected to the selected\n            fields.\n\n        See Also\n        --------\n        get_fields, get_vertex_fields, get_edge_fields\n\n        Examples\n        --------\n        >>> from turicreate import SGraph, Vertex\n        >>> verts = [Vertex(0, attr={'breed': 'labrador', 'age': 5}),\n                     Vertex(1, attr={'breed': 'labrador', 'age': 3}),\n                     Vertex(2, attr={'breed': 'vizsla', 'age': 8})]\n        >>> g = SGraph()\n        >>> g = g.add_vertices(verts)\n        >>> g2 = g.select_fields(fields=['breed'])\n        "
        if type(fields) is str:
            fields = [fields]
        if not isinstance(fields, list) or not all((type(x) is str for x in fields)):
            raise TypeError('"fields" must be a str or list[str]')
        vfields = self.__proxy__.get_vertex_fields()
        efields = self.__proxy__.get_edge_fields()
        selected_vfields = []
        selected_efields = []
        for f in fields:
            found = False
            if f in vfields:
                selected_vfields.append(f)
                found = True
            if f in efields:
                selected_efields.append(f)
                found = True
            if not found:
                raise ValueError("Field '%s' not in graph" % f)
        with cython_context():
            proxy = self.__proxy__
            proxy = proxy.select_vertex_fields(selected_vfields)
            proxy = proxy.select_edge_fields(selected_efields)
            return SGraph(_proxy=proxy)

    def triple_apply(self, triple_apply_fn, mutated_fields, input_fields=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Apply a transform function to each edge and its associated source and\n        target vertices in parallel. Each edge is visited once and in parallel.\n        Modification to vertex data is protected by lock. The effect on the\n        returned SGraph is equivalent to the following pseudocode:\n\n        >>> PARALLEL FOR (source, edge, target) AS triple in G:\n        ...     LOCK (triple.source, triple.target)\n        ...     (source, edge, target) = triple_apply_fn(triple)\n        ...     UNLOCK (triple.source, triple.target)\n        ... END PARALLEL FOR\n\n        Parameters\n        ----------\n        triple_apply_fn : function : (dict, dict, dict) -> (dict, dict, dict)\n            The function to apply to each triple of (source_vertex, edge,\n            target_vertex). This function must take as input a tuple of\n            (source_data, edge_data, target_data) and return a tuple of\n            (new_source_data, new_edge_data, new_target_data). All variables in\n            the both tuples must be of dict type.\n            This can also be a toolkit extension function which is compiled\n            as a native shared library using SDK.\n\n        mutated_fields : list[str] | str\n            Fields that ``triple_apply_fn`` will mutate. Note: columns that are\n            actually mutated by the triple apply function but not specified in\n            ``mutated_fields`` will have undetermined effects.\n\n        input_fields : list[str] | str, optional\n            Fields that ``triple_apply_fn`` will have access to.\n            The default is ``None``, which grants access to all fields.\n            ``mutated_fields`` will always be included in ``input_fields``.\n\n        Returns\n        -------\n        out : SGraph\n            A new SGraph with updated vertex and edge data. Only fields\n            specified in the ``mutated_fields`` parameter are updated.\n\n        Notes\n        -----\n        - ``triple_apply`` does not currently support creating new fields in the\n          lambda function.\n\n        Examples\n        --------\n        Import turicreate and set up the graph.\n\n        >>> edges = turicreate.SFrame({\'source\': range(9), \'dest\': range(1, 10)})\n        >>> g = turicreate.SGraph()\n        >>> g = g.add_edges(edges, src_field=\'source\', dst_field=\'dest\')\n        >>> g.vertices[\'degree\'] = 0\n\n        Define the function to apply to each (source_node, edge, target_node)\n        triple.\n\n        >>> def degree_count_fn (src, edge, dst):\n                src[\'degree\'] += 1\n                dst[\'degree\'] += 1\n                return (src, edge, dst)\n\n        Apply the function to the SGraph.\n\n        >>> g = g.triple_apply(degree_count_fn, mutated_fields=[\'degree\'])\n\n\n        Using native toolkit extension function:\n\n        .. code-block:: c++\n\n            #include <model_server/lib/toolkit_function_macros.hpp>\n            #include <vector>\n\n            using namespace turi;\n            std::vector<variant_type> connected_components_parameterized(\n              std::map<std::string, flexible_type>& src,\n              std::map<std::string, flexible_type>& edge,\n              std::map<std::string, flexible_type>& dst,\n              std::string column) {\n                if (src[column] < dst[column]) dst[column] = src[column];\n                else src[column] = dst[column];\n                return {to_variant(src), to_variant(edge), to_variant(dst)};\n            }\n\n            BEGIN_FUNCTION_REGISTRATION\n            REGISTER_FUNCTION(connected_components_parameterized, "src", "edge", "dst", "column");\n            END_FUNCTION_REGISTRATION\n\n        compiled into example.so\n\n        >>> from example import connected_components_parameterized as cc\n        >>> e = tc.SFrame({\'__src_id\':[1,2,3,4,5], \'__dst_id\':[3,1,2,5,4]})\n        >>> g = tc.SGraph().add_edges(e)\n        >>> g.vertices[\'cid\'] = g.vertices[\'__id\']\n        >>> for i in range(2):\n        ...     g = g.triple_apply(lambda src, edge, dst: cc(src, edge, dst, \'cid\'), [\'cid\'], [\'cid\'])\n        >>> g.vertices[\'cid\']\n        dtype: int\n        Rows: 5\n        [4, 1, 1, 1, 4]\n        '
        assert inspect.isfunction(triple_apply_fn), 'Input must be a function'
        if not (type(mutated_fields) is list or type(mutated_fields) is str):
            raise TypeError('mutated_fields must be str or list of str')
        if not (input_fields is None or type(input_fields) is list or type(input_fields) is str):
            raise TypeError('input_fields must be str or list of str')
        if type(mutated_fields) == str:
            mutated_fields = [mutated_fields]
        if len(mutated_fields) == 0:
            raise ValueError('mutated_fields cannot be empty')
        for f in ['__id', '__src_id', '__dst_id']:
            if f in mutated_fields:
                raise ValueError('mutated_fields cannot contain %s' % f)
        all_fields = self.get_fields()
        if not set(mutated_fields).issubset(set(all_fields)):
            extra_fields = list(set(mutated_fields).difference(set(all_fields)))
            raise ValueError('graph does not contain fields: %s' % str(extra_fields))
        if input_fields is None:
            input_fields = self.get_fields()
        elif type(input_fields) is str:
            input_fields = [input_fields]
        input_fields_set = set(input_fields + mutated_fields)
        input_fields = [x for x in self.get_fields() if x in input_fields_set]
        g = self.select_fields(input_fields)
        nativefn = None
        try:
            from .. import extensions
            nativefn = extensions._build_native_function_call(triple_apply_fn)
        except:
            pass
        if nativefn is not None:
            with cython_context():
                return SGraph(_proxy=g.__proxy__.lambda_triple_apply_native(nativefn, mutated_fields))
        else:
            with cython_context():
                return SGraph(_proxy=g.__proxy__.lambda_triple_apply(triple_apply_fn, mutated_fields))

    def save(self, filename, format='auto'):
        if False:
            print('Hello World!')
        "\n        Save the SGraph to disk. If the graph is saved in binary format, the\n        graph can be re-loaded using the :py:func:`load_sgraph` method.\n        Alternatively, the SGraph can be saved in JSON format for a\n        human-readable and portable representation.\n\n        Parameters\n        ----------\n        filename : string\n            Filename to use when saving the file. It can be either a local or\n            remote url.\n\n        format : {'auto', 'binary', 'json'}, optional\n            File format. If not specified, the format is detected automatically\n            based on the filename. Note that JSON format graphs cannot be\n            re-loaded with :py:func:`load_sgraph`.\n\n        See Also\n        --------\n        load_sgraph\n\n        Examples\n        --------\n        >>> g = turicreate.SGraph()\n        >>> g = g.add_vertices([turicreate.Vertex(i) for i in range(5)])\n\n        Save and load in binary format.\n\n        >>> g.save('mygraph')\n        >>> g2 = turicreate.load_sgraph('mygraph')\n\n        Save in JSON format.\n\n        >>> g.save('mygraph.json', format='json')\n        "
        if format == 'auto':
            if filename.endswith(('.json', '.json.gz')):
                format = 'json'
            else:
                format = 'binary'
        if format not in ['binary', 'json', 'csv']:
            raise ValueError('Invalid format: %s. Supported formats are: %s' % (format, ['binary', 'json', 'csv']))
        with cython_context():
            self.__proxy__.save_graph(_make_internal_url(filename), format)

    def get_neighborhood(self, ids, radius=1, full_subgraph=True):
        if False:
            while True:
                i = 10
        "\n        Retrieve the graph neighborhood around a set of vertices, ignoring edge\n        directions. Note that setting radius greater than two often results in a\n        time-consuming query for a very large subgraph.\n\n        Parameters\n        ----------\n        ids : list [int | float | str]\n            List of target vertex IDs.\n\n        radius : int, optional\n            Radius of the neighborhood. Every vertex in the returned subgraph is\n            reachable from at least one of the target vertices on a path of\n            length no longer than ``radius``. Setting radius larger than 2 may\n            result in a very large subgraph.\n\n        full_subgraph : bool, optional\n            If True, return all edges between vertices in the returned\n            neighborhood. The result is also known as the subgraph induced by\n            the target nodes' neighbors, or the egocentric network for the\n            target nodes. If False, return only edges on paths of length <=\n            ``radius`` from the target node, also known as the reachability\n            graph.\n\n        Returns\n        -------\n        out : Graph\n            The subgraph with the neighborhoods around the target vertices.\n\n        See Also\n        --------\n        get_edges, get_vertices\n\n        References\n        ----------\n        - Marsden, P. (2002) `Egocentric and sociocentric measures of network\n          centrality <http://www.sciencedirect.com/science/article/pii/S03788733\n          02000163>`_.\n        - `Wikipedia - Reachability <http://en.wikipedia.org/wiki/Reachability>`_\n\n        Examples\n        --------\n        >>> sf_edge = turicreate.SFrame({'source': range(9), 'dest': range(1, 10)})\n        >>> g = turicreate.SGraph()\n        >>> g = g.add_edges(sf_edge, src_field='source', dst_field='dest')\n        >>> subgraph = g.get_neighborhood(ids=[1, 7], radius=2,\n                                          full_subgraph=True)\n        "
        verts = ids
        for i in range(radius):
            edges_out = self.get_edges(src_ids=verts)
            edges_in = self.get_edges(dst_ids=verts)
            verts = list(edges_in['__src_id']) + list(edges_in['__dst_id']) + list(edges_out['__src_id']) + list(edges_out['__dst_id'])
            verts = list(set(verts))
        g = SGraph()
        g = g.add_vertices(self.get_vertices(verts), vid_field='__id')
        if full_subgraph is True:
            induced_edge_out = self.get_edges(src_ids=verts)
            induced_edge_in = self.get_edges(dst_ids=verts)
            df_induced = induced_edge_out.append(induced_edge_in)
            df_induced = df_induced.groupby(df_induced.column_names(), {})
            verts_sa = SArray(list(verts))
            edges = df_induced.filter_by(verts_sa, '__src_id')
            edges = edges.filter_by(verts_sa, '__dst_id')
        else:
            path_edges = edges_out.append(edges_in)
            edges = path_edges.groupby(path_edges.column_names(), {})
        g = g.add_edges(edges, src_field='__src_id', dst_field='__dst_id')
        return g

def load_sgraph(filename, format='binary', delimiter='auto'):
    if False:
        i = 10
        return i + 15
    "\n    Load SGraph from text file or previously saved SGraph binary.\n\n    Parameters\n    ----------\n    filename : string\n        Location of the file. Can be a local path or a remote URL.\n\n    format : {'binary', 'snap', 'csv', 'tsv'}, optional\n        Format to of the file to load.\n\n        - 'binary': native graph format obtained from `SGraph.save`.\n        - 'snap': tab or space separated edge list format with comments, used in\n          the `Stanford Network Analysis Platform <http://snap.stanford.edu/snap/>`_.\n        - 'csv': comma-separated edge list without header or comments.\n        - 'tsv': tab-separated edge list without header or comments.\n\n    delimiter : str, optional\n        Specifying the Delimiter used in 'snap', 'csv' or 'tsv' format. Those\n        format has default delimiter, but sometimes it is useful to\n        overwrite the default delimiter.\n\n    Returns\n    -------\n    out : SGraph\n        Loaded SGraph.\n\n    See Also\n    --------\n    SGraph, SGraph.save\n\n    Examples\n    --------\n    >>> g = turicreate.SGraph().add_vertices([turicreate.Vertex(i) for i in range(5)])\n\n    Save and load in binary format.\n\n    >>> g.save('mygraph')\n    >>> g2 = turicreate.load_sgraph('mygraph')\n    "
    if not format in ['binary', 'snap', 'csv', 'tsv']:
        raise ValueError('Invalid format: %s' % format)
    with cython_context():
        g = None
        if format == 'binary':
            proxy = glconnect.get_unity().load_graph(_make_internal_url(filename))
            g = SGraph(_proxy=proxy)
        elif format == 'snap':
            if delimiter == 'auto':
                delimiter = '\t'
            sf = SFrame.read_csv(filename, comment_char='#', delimiter=delimiter, header=False, column_type_hints=int)
            g = SGraph().add_edges(sf, 'X1', 'X2')
        elif format == 'csv':
            if delimiter == 'auto':
                delimiter = ','
            sf = SFrame.read_csv(filename, header=False, delimiter=delimiter)
            g = SGraph().add_edges(sf, 'X1', 'X2')
        elif format == 'tsv':
            if delimiter == 'auto':
                delimiter = '\t'
            sf = SFrame.read_csv(filename, header=False, delimiter=delimiter)
            g = SGraph().add_edges(sf, 'X1', 'X2')
        g.summary()
        return g

def _vertex_list_to_dataframe(ls, id_column_name):
    if False:
        while True:
            i = 10
    '\n    Convert a list of vertices into dataframe.\n    '
    assert HAS_PANDAS, 'Cannot use dataframe because Pandas is not available or version is too low.'
    cols = reduce(set.union, (set(v.attr.keys()) for v in ls))
    df = pd.DataFrame({id_column_name: [v.vid for v in ls]})
    for c in cols:
        df[c] = [v.attr.get(c) for v in ls]
    return df

def _vertex_list_to_sframe(ls, id_column_name):
    if False:
        return 10
    '\n    Convert a list of vertices into an SFrame.\n    '
    sf = SFrame()
    if type(ls) == list:
        cols = reduce(set.union, (set(v.attr.keys()) for v in ls))
        sf[id_column_name] = [v.vid for v in ls]
        for c in cols:
            sf[c] = [v.attr.get(c) for v in ls]
    elif type(ls) == Vertex:
        sf[id_column_name] = [ls.vid]
        for (col, val) in ls.attr.iteritems():
            sf[col] = [val]
    else:
        raise TypeError('Vertices type {} is Not supported.'.format(type(ls)))
    return sf

def _edge_list_to_dataframe(ls, src_column_name, dst_column_name):
    if False:
        while True:
            i = 10
    '\n    Convert a list of edges into dataframe.\n    '
    assert HAS_PANDAS, 'Cannot use dataframe because Pandas is not available or version is too low.'
    cols = reduce(set.union, (set(e.attr.keys()) for e in ls))
    df = pd.DataFrame({src_column_name: [e.src_vid for e in ls], dst_column_name: [e.dst_vid for e in ls]})
    for c in cols:
        df[c] = [e.attr.get(c) for e in ls]
    return df

def _edge_list_to_sframe(ls, src_column_name, dst_column_name):
    if False:
        print('Hello World!')
    '\n    Convert a list of edges into an SFrame.\n    '
    sf = SFrame()
    if type(ls) == list:
        cols = reduce(set.union, (set(v.attr.keys()) for v in ls))
        sf[src_column_name] = [e.src_vid for e in ls]
        sf[dst_column_name] = [e.dst_vid for e in ls]
        for c in cols:
            sf[c] = [e.attr.get(c) for e in ls]
    elif type(ls) == Edge:
        sf[src_column_name] = [ls.src_vid]
        sf[dst_column_name] = [ls.dst_vid]
    else:
        raise TypeError('Edges type {} is Not supported.'.format(type(ls)))
    return sf

def _dataframe_to_vertex_list(df):
    if False:
        for i in range(10):
            print('nop')
    '\n    Convert dataframe into list of vertices, assuming that vertex ids are stored in _VID_COLUMN.\n    '
    cols = df.columns
    if len(cols):
        assert _VID_COLUMN in cols, 'Vertex DataFrame must contain column %s' % _VID_COLUMN
        df = df[cols].T
        ret = [Vertex(None, _series=df[col]) for col in df]
        return ret
    else:
        return []

def _dataframe_to_edge_list(df):
    if False:
        while True:
            i = 10
    '\n    Convert dataframe into list of edges, assuming that source and target ids are stored in _SRC_VID_COLUMN, and _DST_VID_COLUMN respectively.\n    '
    cols = df.columns
    if len(cols):
        assert _SRC_VID_COLUMN in cols, 'Vertex DataFrame must contain column %s' % _SRC_VID_COLUMN
        assert _DST_VID_COLUMN in cols, 'Vertex DataFrame must contain column %s' % _DST_VID_COLUMN
        df = df[cols].T
        ret = [Edge(None, None, _series=df[col]) for col in df]
        return ret
    else:
        return []

def _vertex_data_to_sframe(data, vid_field):
    if False:
        while True:
            i = 10
    "\n    Convert data into a vertex data sframe. Using vid_field to identify the id\n    column. The returned sframe will have id column name '__id'.\n    "
    if isinstance(data, SFrame):
        if vid_field is None and _VID_COLUMN in data.column_names():
            return data
        if vid_field is None:
            raise ValueError('vid_field must be specified for SFrame input')
        data_copy = copy.copy(data)
        data_copy.rename({vid_field: _VID_COLUMN}, inplace=True)
        return data_copy
    if type(data) == Vertex or type(data) == list:
        return _vertex_list_to_sframe(data, '__id')
    elif HAS_PANDAS and type(data) == pd.DataFrame:
        if vid_field is None:
            if data.index.is_unique:
                if not 'index' in data.columns:
                    sf = SFrame(data.reset_index())
                    sf.rename({'index': _VID_COLUMN}, inplace=True)
                    return sf
                else:
                    sf = SFrame(data.reset_index())
                    sf.rename({'level_0': _VID_COLUMN}, inplace=True)
                    return sf
            else:
                raise ValueError('Index of the vertices dataframe is not unique,                         try specifying vid_field name to use a column for vertex ids.')
        else:
            sf = SFrame(data)
            if _VID_COLUMN in sf.column_names():
                raise ValueError('%s reserved vid column name already exists in the SFrame' % _VID_COLUMN)
            sf.rename({vid_field: _VID_COLUMN}, inplace=True)
            return sf
    else:
        raise TypeError('Vertices type %s is Not supported.' % str(type(data)))

def _edge_data_to_sframe(data, src_field, dst_field):
    if False:
        for i in range(10):
            print('nop')
    "\n    Convert data into an edge data sframe. Using src_field and dst_field to\n    identify the source and target id column. The returned sframe will have id\n    column name '__src_id', '__dst_id'\n    "
    if isinstance(data, SFrame):
        if src_field is None and dst_field is None and (_SRC_VID_COLUMN in data.column_names()) and (_DST_VID_COLUMN in data.column_names()):
            return data
        if src_field is None:
            raise ValueError('src_field must be specified for SFrame input')
        if dst_field is None:
            raise ValueError('dst_field must be specified for SFrame input')
        data_copy = copy.copy(data)
        if src_field == _DST_VID_COLUMN and dst_field == _SRC_VID_COLUMN:
            dst_id_column = data_copy[_DST_VID_COLUMN]
            del data_copy[_DST_VID_COLUMN]
            data_copy.rename({_SRC_VID_COLUMN: _DST_VID_COLUMN}, inplace=True)
            data_copy[_SRC_VID_COLUMN] = dst_id_column
        else:
            data_copy.rename({src_field: _SRC_VID_COLUMN, dst_field: _DST_VID_COLUMN}, inplace=True)
        return data_copy
    elif HAS_PANDAS and type(data) == pd.DataFrame:
        if src_field is None:
            raise ValueError('src_field must be specified for Pandas input')
        if dst_field is None:
            raise ValueError('dst_field must be specified for Pandas input')
        sf = SFrame(data)
        if src_field == _DST_VID_COLUMN and dst_field == _SRC_VID_COLUMN:
            dst_id_column = data_copy[_DST_VID_COLUMN]
            del sf[_DST_VID_COLUMN]
            sf.rename({_SRC_VID_COLUMN: _DST_VID_COLUMN}, inplace=True)
            sf[_SRC_VID_COLUMN] = dst_id_column
        else:
            sf.rename({src_field: _SRC_VID_COLUMN, dst_field: _DST_VID_COLUMN}, inplace=True)
        return sf
    elif type(data) == Edge:
        return _edge_list_to_sframe([data], _SRC_VID_COLUMN, _DST_VID_COLUMN)
    elif type(data) == list:
        return _edge_list_to_sframe(data, _SRC_VID_COLUMN, _DST_VID_COLUMN)
    else:
        raise TypeError('Edges type %s is Not supported.' % str(type(data)))
GFrame.__name__ = SFrame.__name__
GFrame.__module__ = SFrame.__module__