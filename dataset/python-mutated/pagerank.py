from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
from turicreate.data_structures.sgraph import SGraph as _SGraph
from turicreate.toolkits.graph_analytics._model_base import GraphAnalyticsModel as _ModelBase

class PagerankModel(_ModelBase):
    """
    A PageRankModel object contains the pagerank value for each vertex.
    The pagerank value characterizes the importance of a vertex
    in the graph using the following recursive definition:

        .. math::
          pr(i) =  reset_probability + (1-reset_probability) \\sum_{j\\in N(i)} pr(j) / out_degree(j)

    where :math:`N(i)` is the set containing all vertices :math:`j` such that
    there is an edge going from :math:`j` to :math:`i`. Self edges (i.e., edges
    where the source vertex is the same as the destination vertex) and repeated
    edges (i.e., multiple edges where the source vertices are the same and the
    destination vertices are the same) are treated like normal edges in the
    above recursion.

    Currently, edge weights are not taken into account when computing the
    PageRank.

    Below is a list of queryable fields for this model:

    +-------------------+-----------------------------------------------------------+
    | Field             | Description                                               |
    +===================+===========================================================+
    | reset_probability | The probability of random jumps to any node in the graph  |
    +-------------------+-----------------------------------------------------------+
    | graph             | A new SGraph with the pagerank as a vertex property       |
    +-------------------+-----------------------------------------------------------+
    | delta             | Total changes in pagerank during the last iteration       |
    |                   | (the L1 norm of the changes)                              |
    +-------------------+-----------------------------------------------------------+
    | pagerank          | An SFrame with each vertex's pagerank                     |
    +-------------------+-----------------------------------------------------------+
    | num_iterations    | Number of iterations                                      |
    +-------------------+-----------------------------------------------------------+
    | threshold         | The convergence threshold in L1 norm                      |
    +-------------------+-----------------------------------------------------------+
    | training_time     | Total training time of the model                          |
    +-------------------+-----------------------------------------------------------+
    | max_iterations    | The maximum number of iterations to run                   |
    +-------------------+-----------------------------------------------------------+


    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.pagerank.create` to create an instance
    of this model. A detailed list of parameter options and code samples
    are available in the documentation for the create function.

    See Also
    --------
    create
    """

    def __init__(self, model):
        if False:
            while True:
                i = 10
        '__init__(self)'
        self.__proxy__ = model

    def _result_fields(self):
        if False:
            return 10
        ret = super(PagerankModel, self)._result_fields()
        ret['vertex pagerank'] = 'SFrame. See m.pagerank'
        ret['change in last iteration (L1 norm)'] = self.delta
        return ret

    def _metric_fields(self):
        if False:
            return 10
        ret = super(PagerankModel, self)._metric_fields()
        ret['number of iterations'] = 'num_iterations'
        return ret

    def _setting_fields(self):
        if False:
            return 10
        ret = super(PagerankModel, self)._setting_fields()
        ret['probability of random jumps to any node in the graph'] = 'reset_probability'
        ret['convergence threshold (L1 norm)'] = 'threshold'
        ret['maximum number of iterations'] = 'max_iterations'
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
        return 'pagerank'

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

def create(graph, reset_probability=0.15, threshold=0.01, max_iterations=20, _single_precision=False, _distributed='auto', verbose=True):
    if False:
        return 10
    "\n    Compute the PageRank for each vertex in the graph. Return a model object\n    with total PageRank as well as the PageRank value for each vertex in the\n    graph.\n\n    Parameters\n    ----------\n    graph : SGraph\n        The graph on which to compute the pagerank value.\n\n    reset_probability : float, optional\n        Probability that a random surfer jumps to an arbitrary page.\n\n    threshold : float, optional\n        Threshold for convergence, measured in the L1 norm\n        (the sum of absolute value) of the delta of each vertex's\n        pagerank value.\n\n    max_iterations : int, optional\n        The maximum number of iterations to run.\n\n    _single_precision : bool, optional\n        If true, running pagerank in single precision. The resulting\n        pagerank values may not be accurate for large graph, but\n        should run faster and use less memory.\n\n    _distributed : distributed environment, internal\n\n    verbose : bool, optional\n        If True, print progress updates.\n\n\n    Returns\n    -------\n    out : PagerankModel\n\n    References\n    ----------\n    - `Wikipedia - PageRank <http://en.wikipedia.org/wiki/PageRank>`_\n    - Page, L., et al. (1998) `The PageRank Citation Ranking: Bringing Order to\n      the Web <http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf>`_.\n\n    Examples\n    --------\n    If given an :class:`~turicreate.SGraph` ``g``, we can create\n    a :class:`~turicreate.pagerank.PageRankModel` as follows:\n\n    >>> g = turicreate.load_sgraph('http://snap.stanford.edu/data/email-Enron.txt.gz', format='snap')\n    >>> pr = turicreate.pagerank.create(g)\n\n    We can obtain the page rank corresponding to each vertex in the graph ``g``\n    using:\n\n    >>> pr_out = pr['pagerank']     # SFrame\n\n    We can add the new pagerank field to the original graph g using:\n\n    >>> g.vertices['pagerank'] = pr['graph'].vertices['pagerank']\n\n    Note that the task above does not require a join because the vertex\n    ordering is preserved through ``create()``.\n\n    See Also\n    --------\n    PagerankModel\n    "
    from turicreate._cython.cy_server import QuietProgress
    if not isinstance(graph, _SGraph):
        raise TypeError('graph input must be a SGraph object.')
    opts = {'threshold': threshold, 'reset_probability': reset_probability, 'max_iterations': max_iterations, 'single_precision': _single_precision, 'graph': graph.__proxy__}
    with QuietProgress(verbose):
        params = _tc.extensions._toolkits.graph.pagerank.create(opts)
    model = params['model']
    return PagerankModel(model)