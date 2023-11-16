from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
from turicreate.data_structures.sgraph import SGraph as _SGraph
from turicreate.toolkits.graph_analytics._model_base import GraphAnalyticsModel as _ModelBase

class KcoreModel(_ModelBase):
    """
    A KcoreModel object contains a core ID for each vertex, and the total
    number of cores in the graph.

    The core ID of a vertex is a measure of its global centrality.

    The algorithms iteratively remove vertices that has less than :math:`k`
    neighbors **recursively**. The algorithm guarantees that at iteration
    :math:`k+1`, all vertices left in the graph will have at least :math:`k+1`
    neighbors.  The vertices removed at iteration :math:`k` is assigned with a
    core ID equal to :math:`k`.

    Below is a list of queryable fields for this model:

    +---------------+----------------------------------------------------+
    | Field         | Description                                        |
    +===============+====================================================+
    | core_id       | An SFrame with each vertex's core id               |
    +---------------+----------------------------------------------------+
    | graph         | A new SGraph with the core id as a vertex property |
    +---------------+----------------------------------------------------+
    | kmax          | The maximum core id assigned to any vertex         |
    +---------------+----------------------------------------------------+
    | kmin          | The minimum core id assigned to any vertex         |
    +---------------+----------------------------------------------------+
    | training_time | Total training time of the model                   |
    +---------------+----------------------------------------------------+

    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.kcore.create` to create an instance
    of this model. A detailed list of parameter options and code samples
    are available in the documentation for the create function.

    See Also
    --------
    create
    """

    def __init__(self, model):
        if False:
            return 10
        '__init__(self)'
        self.__proxy__ = model
        self.__model_name__ = self.__class__._native_name()

    def _result_fields(self):
        if False:
            return 10
        ret = super(KcoreModel, self)._result_fields()
        ret['vertex core id'] = "SFrame. See m['core_id']"
        return ret

    def _setting_fields(self):
        if False:
            for i in range(10):
                print('nop')
        ret = super(KcoreModel, self)._setting_fields()
        ret['minimum core id assigned to any vertex'] = 'kmin'
        ret['maximum core id assigned to any vertex '] = 'kmax'
        return ret

    def _get_version(self):
        if False:
            for i in range(10):
                print('nop')
        return 0

    @classmethod
    def _native_name(cls):
        if False:
            print('Hello World!')
        return 'kcore'

    def _get_native_state(self):
        if False:
            return 10
        return {'model': self.__proxy__}

    @classmethod
    def _load_version(cls, state, version):
        if False:
            return 10
        assert version == 0
        return cls(state['model'])

def create(graph, kmin=0, kmax=10, verbose=True):
    if False:
        i = 10
        return i + 15
    "\n    Compute the K-core decomposition of the graph. Return a model object with\n    total number of cores as well as the core id for each vertex in the graph.\n\n    Parameters\n    ----------\n    graph : SGraph\n        The graph on which to compute the k-core decomposition.\n\n    kmin : int, optional\n        Minimum core id. Vertices having smaller core id than `kmin` will be\n        assigned with core_id = `kmin`.\n\n    kmax : int, optional\n        Maximum core id. Vertices having larger core id than `kmax` will be\n        assigned with core_id=`kmax`.\n\n    verbose : bool, optional\n        If True, print progress updates.\n\n    Returns\n    -------\n    out : KcoreModel\n\n    References\n    ----------\n    - Alvarez-Hamelin, J.I., et al. (2005) `K-Core Decomposition: A Tool for the\n      Visualization of Large Networks <http://arxiv.org/abs/cs/0504107>`_.\n\n    Examples\n    --------\n    If given an :class:`~turicreate.SGraph` ``g``, we can create\n    a :class:`~turicreate.kcore.KcoreModel` as follows:\n\n    >>> g = turicreate.load_sgraph('http://snap.stanford.edu/data/email-Enron.txt.gz', format='snap')\n    >>> kc = turicreate.kcore.create(g)\n\n    We can obtain the ``core id`` corresponding to each vertex in the graph\n    ``g`` using:\n\n    >>> kcore_id = kc['core_id']     # SFrame\n\n    We can add the new core id field to the original graph g using:\n\n    >>> g.vertices['core_id'] = kc['graph'].vertices['core_id']\n\n    Note that the task above does not require a join because the vertex\n    ordering is preserved through ``create()``.\n\n    See Also\n    --------\n    KcoreModel\n    "
    from turicreate._cython.cy_server import QuietProgress
    if not isinstance(graph, _SGraph):
        raise TypeError('graph input must be a SGraph object.')
    opts = {'graph': graph.__proxy__, 'kmin': kmin, 'kmax': kmax}
    with QuietProgress(verbose):
        params = _tc.extensions._toolkits.graph.kcore.create(opts)
    return KcoreModel(params['model'])