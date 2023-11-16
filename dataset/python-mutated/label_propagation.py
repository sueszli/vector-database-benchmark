from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _tc
from turicreate.data_structures.sgraph import SGraph as _SGraph
from turicreate.toolkits.graph_analytics._model_base import GraphAnalyticsModel as _ModelBase
from turicreate.util import _raise_error_if_not_of_type

class LabelPropagationModel(_ModelBase):
    """
    A LabelPropagationModel computes the probability of each class label
    for each unlabeled vertex.

    For each labeled vertices, the probability for class k is fixed to:

        .. math::
          Pr_i(label=k) = I(label[i] == k)

    where :math:`I()` is the indicator function.

    For all unlabeled vertices, the probability for each class k is computed
    from applying the following update iteratively:

        .. math::
          Pr_i(label=k) = Pr_i(label=k) * W_0 + \\sum_{j\\in N(i)} Pr_j(label=k) * W(j,i)

          Pr_i = Normalize(Pr_i)

    where :math:`N(i)` is the set containing all vertices :math:`j` such that
    there is an edge going from :math:`j` to :math:`i`. :math:`W(j,i)` is
    the edge weight from :math:`j` to :math:`i`, and :math:`W_0` is the
    weight for self edge.

    In the above equation, the first term is the probability
    of keeping the label from the previous iteration, and the second term
    is the probability of transition to a neighbor's label.

    Repeated edges (i.e., multiple edges where the source vertices are the same and the
    destination vertices are the same) are treated like normal edges in the
    above recursion.

    By default, the label propagates from source to target. But if `undirected`
    is set to true in :func:`turicreate.label_propagation.create`, then the label
    propagates in both directions for each edge.

    Below is a list of queryable fields for this model:

    +-------------------+-----------------------------------------------------------+
    | Field             | Description                                               |
    +===================+===========================================================+
    | labels            | An SFrame with label probability for each vertex          |
    +-------------------+-----------------------------------------------------------+
    | graph             | A new SGraph with label probability as vertex properties  |
    +-------------------+-----------------------------------------------------------+
    | delta             | Average changes in label probability during the last      |
    |                   | iteration (avg. of the L2 norm of the changes)            |
    +-------------------+-----------------------------------------------------------+
    | num_iterations    | Number of iterations                                      |
    +-------------------+-----------------------------------------------------------+
    | training_time     | Total training time of the model                          |
    +-------------------+-----------------------------------------------------------+
    | threshold         | The convergence threshold in average L2 norm              |
    +-------------------+-----------------------------------------------------------+
    | self_weight       | The weight for self edge                                  |
    +-------------------+-----------------------------------------------------------+
    | weight_field      | The edge weight field id                                  |
    +-------------------+-----------------------------------------------------------+
    | label_field       | The vertex label field id                                 |
    +-------------------+-----------------------------------------------------------+
    | undirected        | Treat edge as undirected                                  |
    +-------------------+-----------------------------------------------------------+

    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.label_propagation.create` to create an instance
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

    def _result_fields(self):
        if False:
            print('Hello World!')
        ret = super(LabelPropagationModel, self)._result_fields()
        ret['vertex label probability'] = "SFrame. See m['labels']"
        ret['change in last iteration (avg. of L2)'] = self['delta']
        return ret

    def _metric_fields(self):
        if False:
            print('Hello World!')
        ret = super(LabelPropagationModel, self)._metric_fields()
        ret['number of iterations'] = 'num_iterations'
        return ret

    def _setting_fields(self):
        if False:
            print('Hello World!')
        ret = super(LabelPropagationModel, self)._setting_fields()
        ret['convergence threshold (avg. of L2 norm)'] = 'threshold'
        ret['treated edge as undirected'] = 'undirected'
        ret['weight for self edge'] = 'self_weight'
        ret['edge weight field id'] = 'weight_field'
        ret['vertex label field id'] = 'label_field'
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
        return 'label_propagation'

    def _get_native_state(self):
        if False:
            i = 10
            return i + 15
        return {'model': self.__proxy__}

    @classmethod
    def _load_version(cls, state, version):
        if False:
            while True:
                i = 10
        assert version == 0
        return cls(state['model'])

def create(graph, label_field, threshold=0.001, weight_field='', self_weight=1.0, undirected=False, max_iterations=None, _single_precision=False, _distributed='auto', verbose=True):
    if False:
        print('Hello World!')
    '\n    Given a weighted graph with observed class labels of a subset of vertices,\n    infer the label probability for the unobserved vertices using the\n    "label propagation" algorithm.\n\n    The algorithm iteratively updates the label probability of current vertex\n    as a weighted sum of label probability of self and the neighboring vertices\n    until converge.  See\n    :class:`turicreate.label_propagation.LabelPropagationModel` for the details\n    of the algorithm.\n\n    Notes: label propagation works well with small number of labels, i.e. binary\n    labels, or less than 1000 classes. The toolkit will throw error\n    if the number of classes exceeds the maximum value (1000).\n\n    Parameters\n    ----------\n    graph : SGraph\n        The graph on which to compute the label propagation.\n\n    label_field: str\n        Vertex field storing the initial vertex labels. The values in\n        must be [0, num_classes). None values indicate unobserved vertex labels.\n\n    threshold : float, optional\n        Threshold for convergence, measured in the average L2 norm\n        (the sum of squared values) of the delta of each vertex\'s\n        label probability vector.\n\n    max_iterations: int, optional\n        The max number of iterations to run. Default is unlimited.\n        If set, the algorithm terminates when either max_iterations\n        or convergence threshold is reached.\n\n    weight_field: str, optional\n        Vertex field for edge weight. If empty, all edges are assumed\n        to have unit weight.\n\n    self_weight: float, optional\n        The weight for self edge.\n\n    undirected: bool, optional\n        If true, treat each edge as undirected, and propagates label in\n        both directions.\n\n    _single_precision : bool, optional\n        If true, running label propagation in single precision. The resulting\n        probability values may less accurate, but should run faster\n        and use less memory.\n\n    _distributed : distributed environment, internal\n\n    verbose : bool, optional\n        If True, print progress updates.\n\n    Returns\n    -------\n    out : LabelPropagationModel\n\n    References\n    ----------\n    - Zhu, X., & Ghahramani, Z. (2002). `Learning from labeled and unlabeled data\n      with label propagation <http://www.cs.cmu.edu/~zhuxj/pub/CMU-CALD-02-107.pdf>`_.\n\n    Examples\n    --------\n    If given an :class:`~turicreate.SGraph` ``g``, we can create\n    a :class:`~turicreate.label_propagation.LabelPropagationModel` as follows:\n\n    >>> g = turicreate.load_sgraph(\'http://snap.stanford.edu/data/email-Enron.txt.gz\',\n    ...                         format=\'snap\')\n    # Initialize random classes for a subset of vertices\n    # Leave the unobserved vertices with None label.\n    >>> import random\n    >>> def init_label(vid):\n    ...     x = random.random()\n    ...     if x < 0.2:\n    ...         return 0\n    ...     elif x > 0.9:\n    ...         return 1\n    ...     else:\n    ...         return None\n    >>> g.vertices[\'label\'] = g.vertices[\'__id\'].apply(init_label, int)\n    >>> m = turicreate.label_propagation.create(g, label_field=\'label\')\n\n    We can obtain for each vertex the predicted label and the probability of\n    each label in the graph ``g`` using:\n\n    >>> labels = m[\'labels\']     # SFrame\n    >>> labels\n    +------+-------+-----------------+-------------------+----------------+\n    | __id | label | predicted_label |         P0        |       P1       |\n    +------+-------+-----------------+-------------------+----------------+\n    |  5   |   1   |        1        |        0.0        |      1.0       |\n    |  7   |  None |        0        |    0.8213214997   |  0.1786785003  |\n    |  8   |  None |        1        | 5.96046447754e-08 | 0.999999940395 |\n    |  10  |  None |        0        |   0.534984718273  | 0.465015281727 |\n    |  27  |  None |        0        |   0.752801638549  | 0.247198361451 |\n    |  29  |  None |        1        | 5.96046447754e-08 | 0.999999940395 |\n    |  33  |  None |        1        | 5.96046447754e-08 | 0.999999940395 |\n    |  47  |   0   |        0        |        1.0        |      0.0       |\n    |  50  |  None |        0        |   0.788279032657  | 0.211720967343 |\n    |  52  |  None |        0        |   0.666666666667  | 0.333333333333 |\n    +------+-------+-----------------+-------------------+----------------+\n    [36692 rows x 5 columns]\n\n    See Also\n    --------\n    LabelPropagationModel\n    '
    from turicreate._cython.cy_server import QuietProgress
    _raise_error_if_not_of_type(label_field, str)
    _raise_error_if_not_of_type(weight_field, str)
    if not isinstance(graph, _SGraph):
        raise TypeError('graph input must be a SGraph object.')
    if graph.vertices[label_field].dtype != int:
        raise TypeError('label_field %s must be integer typed.' % label_field)
    opts = {'label_field': label_field, 'threshold': threshold, 'weight_field': weight_field, 'self_weight': self_weight, 'undirected': undirected, 'max_iterations': max_iterations, 'single_precision': _single_precision, 'graph': graph.__proxy__}
    with QuietProgress(verbose):
        params = _tc.extensions._toolkits.graph.label_propagation.create(opts)
    model = params['model']
    return LabelPropagationModel(model)