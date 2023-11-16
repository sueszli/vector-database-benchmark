from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
from turicreate.data_structures.sgraph import SGraph as _SGraph
from turicreate.toolkits.graph_analytics._model_base import GraphAnalyticsModel as _ModelBase

class ConnectedComponentsModel(_ModelBase):
    """
    A ConnectedComponentsModel object contains the component ID for each vertex
    and the total number of weakly connected components in the graph.

    A weakly connected component is a maximal set of vertices such that there
    exists an undirected path between any two vertices in the set.

    Below is a list of queryable fields for this model:

    +----------------+-----------------------------------------------------+
    | Field          | Description                                         |
    +================+=====================================================+
    | graph          | A new SGraph with the color id as a vertex property |
    +----------------+-----------------------------------------------------+
    | training_time  | Total training time of the model                    |
    +----------------+-----------------------------------------------------+
    | component_size | An SFrame with the size of each component           |
    +----------------+-----------------------------------------------------+
    | component_id   | An SFrame with each vertex's component id           |
    +----------------+-----------------------------------------------------+

    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.connected_components.create` to create an instance
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

    def _get_version(self):
        if False:
            i = 10
            return i + 15
        return 0

    @classmethod
    def _native_name(cls):
        if False:
            return 10
        return 'connected_components'

    def _get_native_state(self):
        if False:
            print('Hello World!')
        return {'model': self.__proxy__}

    @classmethod
    def _load_version(cls, state, version):
        if False:
            i = 10
            return i + 15
        assert version == 0
        return cls(state['model'])

    def _result_fields(self):
        if False:
            for i in range(10):
                print('nop')
        ret = super(ConnectedComponentsModel, self)._result_fields()
        ret['number of connected components'] = len(self['component_size'])
        ret['component size'] = "SFrame. See m['component_size']"
        ret['vertex component id'] = "SFrame. See m['component_id']"
        return ret

def create(graph, verbose=True):
    if False:
        i = 10
        return i + 15
    "\n    Compute the number of weakly connected components in the graph. Return a\n    model object with total number of weakly connected components as well as the\n    component ID for each vertex in the graph.\n\n    Parameters\n    ----------\n    graph : SGraph\n        The graph on which to compute the triangle counts.\n\n    verbose : bool, optional\n        If True, print progress updates.\n\n    Returns\n    -------\n    out : ConnectedComponentsModel\n\n    References\n    ----------\n    - `Mathworld Wolfram - Weakly Connected Component\n      <http://mathworld.wolfram.com/WeaklyConnectedComponent.html>`_\n\n    Examples\n    --------\n    If given an :class:`~turicreate.SGraph` ``g``, we can create\n    a :class:`~turicreate.connected_components.ConnectedComponentsModel` as\n    follows:\n\n    >>> g = turicreate.load_sgraph('http://snap.stanford.edu/data/email-Enron.txt.gz', format='snap')\n    >>> cc = turicreate.connected_components.create(g)\n    >>> cc.summary()\n\n    We can obtain the ``component id`` corresponding to each vertex in the\n    graph ``g`` as follows:\n\n    >>> cc_ids = cc['component_id']  # SFrame\n\n    We can obtain a graph with additional information about the ``component\n    id`` corresponding to each vertex as follows:\n\n    >>> cc_graph = cc['graph']      # SGraph\n\n    We can add the new component_id field to the original graph g using:\n\n    >>> g.vertices['component_id'] = cc['graph'].vertices['component_id']\n\n    Note that the task above does not require a join because the vertex\n    ordering is preserved through ``create()``.\n\n\n    See Also\n    --------\n    ConnectedComponentsModel\n    "
    from turicreate._cython.cy_server import QuietProgress
    if not isinstance(graph, _SGraph):
        raise TypeError('"graph" input must be a SGraph object.')
    with QuietProgress(verbose):
        params = _tc.extensions._toolkits.graph.connected_components.create({'graph': graph.__proxy__})
    return ConnectedComponentsModel(params['model'])