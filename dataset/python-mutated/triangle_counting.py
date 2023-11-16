from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
from turicreate.data_structures.sgraph import SGraph as _SGraph
from turicreate.toolkits.graph_analytics._model_base import GraphAnalyticsModel as _ModelBase

class TriangleCountingModel(_ModelBase):
    """
    Model object containing the triangle count for each vertex, and the total
    number of triangles. The model ignores the edge directions in that
    it assumes there are no multiple edges between
    the same source ang target pair and ignores bidirectional edges.

    The triangle count of individual vertex characterizes the importance of the
    vertex in its neighborhood. The total number of triangles characterizes the
    density of the graph. It can also be calculated using

    >>> m['triangle_count']['triangle_count'].sum() / 3.

    Below is a list of queryable fields for this model:

    +---------------+------------------------------------------------------------+
    | Field         | Description                                                |
    +===============+============================================================+
    | triangle_count| An SFrame with each vertex's id and triangle count         |
    +---------------+------------------------------------------------------------+
    | num_triangles | Total number of triangles in the graph                     |
    +---------------+------------------------------------------------------------+
    | graph         | A new SGraph with the triangle count as a vertex property  |
    +---------------+------------------------------------------------------------+
    | training_time | Total training time of the model                           |
    +---------------+------------------------------------------------------------+

    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.triangle_counting.create` to create an instance
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

    def _result_fields(self):
        if False:
            i = 10
            return i + 15
        ret = super(TriangleCountingModel, self)._result_fields()
        ret['total number of triangles'] = self.num_triangles
        ret['vertex triangle count'] = 'SFrame. See m.triangle_count'
        return ret

    def _get_version(self):
        if False:
            print('Hello World!')
        return 0

    @classmethod
    def _native_name(cls):
        if False:
            i = 10
            return i + 15
        return 'triangle_count'

    def _get_native_state(self):
        if False:
            while True:
                i = 10
        return {'model': self.__proxy__}

    @classmethod
    def _load_version(cls, state, version):
        if False:
            i = 10
            return i + 15
        assert version == 0
        return cls(state['model'])

def create(graph, verbose=True):
    if False:
        i = 10
        return i + 15
    '\n    Compute the number of triangles each vertex belongs to, ignoring edge\n    directions. A triangle is a complete subgraph with only three vertices.\n    Return a model object with total number of triangles as well as the triangle\n    counts for each vertex in the graph.\n\n    Parameters\n    ----------\n    graph : SGraph\n        The graph on which to compute triangle counts.\n\n    verbose : bool, optional\n        If True, print progress updates.\n\n    Returns\n    -------\n    out : TriangleCountingModel\n\n    References\n    ----------\n    - T. Schank. (2007) `Algorithmic Aspects of Triangle-Based Network Analysis\n      <http://digbib.ubka.uni-karlsruhe.de/volltexte/documents/4541>`_.\n\n    Examples\n    --------\n    If given an :class:`~turicreate.SGraph` ``g``, we can create a\n    :class:`~turicreate.triangle_counting.TriangleCountingModel` as follows:\n\n    >>> g =\n    >>> turicreate.load_sgraph(\'http://snap.stanford.edu/data/email-Enron.txt.gz\',\n            >>> format=\'snap\') tc = turicreate.triangle_counting.create(g)\n\n    We can obtain the number of triangles that each vertex in the graph ``g``\n    is present in:\n\n    >>> tc_out = tc[\'triangle_count\']  # SFrame\n\n    We can add the new "triangle_count" field to the original graph g using:\n\n    >>> g.vertices[\'triangle_count\'] = tc[\'graph\'].vertices[\'triangle_count\']\n\n    Note that the task above does not require a join because the vertex\n    ordering is preserved through ``create()``.\n\n    See Also\n    --------\n    TriangleCountingModel\n    '
    from turicreate._cython.cy_server import QuietProgress
    if not isinstance(graph, _SGraph):
        raise TypeError('graph input must be a SGraph object.')
    with QuietProgress(verbose):
        params = _tc.extensions._toolkits.graph.triangle_counting.create({'graph': graph.__proxy__})
    return TriangleCountingModel(params['model'])