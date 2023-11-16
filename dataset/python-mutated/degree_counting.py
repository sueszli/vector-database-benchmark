from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
from turicreate.data_structures.sgraph import SGraph as _SGraph
from turicreate.toolkits.graph_analytics._model_base import GraphAnalyticsModel as _ModelBase

class DegreeCountingModel(_ModelBase):
    """
    Model object containing the in degree, out degree and total degree for each vertex,

    Below is a list of queryable fields for this model:

    +---------------+------------------------------------------------------------+
    | Field         | Description                                                |
    +===============+============================================================+
    | graph         | A new SGraph with the degree counts as vertex properties   |
    +---------------+------------------------------------------------------------+
    | training_time | Total training time of the model                           |
    +---------------+------------------------------------------------------------+

    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.degree_counting.create` to create an instance
    of this model. A detailed list of parameter options and code samples
    are available in the documentation for the create function.

    See Also
    --------
    create
    """

    def __init__(self, model):
        if False:
            i = 10
            return i + 15
        '__init__(self)'
        self.__proxy__ = model

    def _get_version(self):
        if False:
            for i in range(10):
                print('nop')
        return 0

    @classmethod
    def _native_name(cls):
        if False:
            while True:
                i = 10
        return 'degree_count'

    def _get_native_state(self):
        if False:
            while True:
                i = 10
        return {'model': self.__proxy__}

    @classmethod
    def _load_version(cls, state, version):
        if False:
            while True:
                i = 10
        assert version == 0
        return cls(state['model'])

def create(graph, verbose=True):
    if False:
        while True:
            i = 10
    "\n    Compute the in degree, out degree and total degree of each vertex.\n\n    Parameters\n    ----------\n    graph : SGraph\n        The graph on which to compute degree counts.\n\n    verbose : bool, optional\n        If True, print progress updates.\n\n    Returns\n    -------\n    out : DegreeCountingModel\n\n    Examples\n    --------\n    If given an :class:`~turicreate.SGraph` ``g``, we can create\n    a :class:`~turicreate.degree_counting.DegreeCountingModel` as follows:\n\n    >>> g = turicreate.load_sgraph('http://snap.stanford.edu/data/web-Google.txt.gz',\n    ...                         format='snap')\n    >>> m = turicreate.degree_counting.create(g)\n    >>> g2 = m['graph']\n    >>> g2\n    SGraph({'num_edges': 5105039, 'num_vertices': 875713})\n    Vertex Fields:['__id', 'in_degree', 'out_degree', 'total_degree']\n    Edge Fields:['__src_id', '__dst_id']\n\n    >>> g2.vertices.head(5)\n    Columns:\n        __id\tint\n        in_degree\tint\n        out_degree\tint\n        total_degree\tint\n    <BLANKLINE>\n    Rows: 5\n    <BLANKLINE>\n    Data:\n    +------+-----------+------------+--------------+\n    | __id | in_degree | out_degree | total_degree |\n    +------+-----------+------------+--------------+\n    |  5   |     15    |     7      |      22      |\n    |  7   |     3     |     16     |      19      |\n    |  8   |     1     |     2      |      3       |\n    |  10  |     13    |     11     |      24      |\n    |  27  |     19    |     16     |      35      |\n    +------+-----------+------------+--------------+\n\n    See Also\n    --------\n    DegreeCountingModel\n    "
    from turicreate._cython.cy_server import QuietProgress
    if not isinstance(graph, _SGraph):
        raise TypeError('"graph" input must be a SGraph object.')
    with QuietProgress(verbose):
        params = _tc.extensions._toolkits.graph.degree_count.create({'graph': graph.__proxy__})
    return DegreeCountingModel(params['model'])