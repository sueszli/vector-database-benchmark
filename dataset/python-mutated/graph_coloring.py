from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
from turicreate.data_structures.sgraph import SGraph as _SGraph
from turicreate.toolkits.graph_analytics._model_base import GraphAnalyticsModel as _ModelBase

class GraphColoringModel(_ModelBase):
    """
    A GraphColoringModel object contains color ID assignments for each vertex
    and the total number of colors used in coloring the entire graph.

    The coloring is the result of a greedy algorithm and therefore is not
    optimal.  Finding optimal coloring is in fact NP-complete.

    Below is a list of queryable fields for this model:

    +----------------+-----------------------------------------------------+
    | Field          | Description                                         |
    +================+=====================================================+
    | graph          | A new SGraph with the color id as a vertex property |
    +----------------+-----------------------------------------------------+
    | training_time  | Total training time of the model                    |
    +----------------+-----------------------------------------------------+
    | num_colors     | Number of colors in the graph                       |
    +----------------+-----------------------------------------------------+

    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.graph_coloring.create` to create an instance
    of this model. A detailed list of parameter options and code samples
    are available in the documentation for the create function.


    See Also
    --------
    create
    """

    def __init__(self, model):
        if False:
            for i in range(10):
                print('nop')
        '__init__(self)'
        self.__proxy__ = model
        self.__model_name__ = self.__class__._native_name()

    def _result_fields(self):
        if False:
            return 10
        ret = super(GraphColoringModel, self)._result_fields()
        ret['number of colors in the graph'] = self.num_colors
        ret['vertex color id'] = 'SFrame. See m.color_id'
        return ret

    def _get_version(self):
        if False:
            while True:
                i = 10
        return 0

    @classmethod
    def _native_name(cls):
        if False:
            print('Hello World!')
        return 'graph_coloring'

    def _get_native_state(self):
        if False:
            for i in range(10):
                print('nop')
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
        print('Hello World!')
    "\n    Compute the graph coloring. Assign a color to each vertex such that no\n    adjacent vertices have the same color. Return a model object with total\n    number of colors used as well as the color ID for each vertex in the graph.\n    This algorithm is greedy and is not guaranteed to find the **minimum** graph\n    coloring. It is also not deterministic, so successive runs may return\n    different answers.\n\n    Parameters\n    ----------\n    graph : SGraph\n        The graph on which to compute the coloring.\n\n    verbose : bool, optional\n        If True, print progress updates.\n\n    Returns\n    -------\n    out : GraphColoringModel\n\n    References\n    ----------\n    - `Wikipedia - graph coloring <http://en.wikipedia.org/wiki/Graph_coloring>`_\n\n    Examples\n    --------\n    If given an :class:`~turicreate.SGraph` ``g``, we can create\n    a :class:`~turicreate.graph_coloring.GraphColoringModel` as follows:\n\n    >>> g = turicreate.load_sgraph('http://snap.stanford.edu/data/email-Enron.txt.gz', format='snap')\n    >>> gc = turicreate.graph_coloring.create(g)\n\n    We can obtain the ``color id`` corresponding to each vertex in the graph ``g``\n    as follows:\n\n    >>> color_id = gc['color_id']  # SFrame\n\n    We can obtain the total number of colors required to color the graph ``g``\n    as follows:\n\n    >>> num_colors = gc['num_colors']\n\n    See Also\n    --------\n    GraphColoringModel\n    "
    from turicreate._cython.cy_server import QuietProgress
    if not isinstance(graph, _SGraph):
        raise TypeError('graph input must be a SGraph object.')
    with QuietProgress(verbose):
        params = _tc.extensions._toolkits.graph.graph_coloring.create({'graph': graph.__proxy__})
    return GraphColoringModel(params['model'])