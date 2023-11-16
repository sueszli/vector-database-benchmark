"""
Methods for using factorization-based models as a recommender system.  See
turicreate.recommender.factorization_recommender.create for additional documentation.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from turicreate.toolkits._model import _get_default_options_wrapper
import turicreate as _turicreate
from turicreate.toolkits.recommender.util import _Recommender
from turicreate.data_structures.sframe import SFrame as _SFrame

def create(observation_data, user_id='user_id', item_id='item_id', target=None, user_data=None, item_data=None, num_factors=8, regularization=1e-08, linear_regularization=1e-10, side_data_factorization=True, nmf=False, binary_target=False, max_iterations=50, sgd_step_size=0, random_seed=0, solver='auto', verbose=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Create a FactorizationRecommender that learns latent factors for each\n    user and item and uses them to make rating predictions. This includes\n    both standard matrix factorization as well as factorization machines models\n    (in the situation where side data is available for users and/or items).\n\n    Parameters\n    ----------\n    observation_data : SFrame\n        The dataset to use for training the model. It must contain a column of\n        user ids and a column of item ids. Each row represents an observed\n        interaction between the user and the item.  The (user, item) pairs\n        are stored with the model so that they can later be excluded from\n        recommendations if desired. It can optionally contain a target ratings\n        column. All other columns are interpreted by the underlying model as\n        side features for the observations.\n\n        The user id and item id columns must be of type \'int\' or \'str\'. The\n        target column must be of type \'int\' or \'float\'.\n\n    user_id : string, optional\n        The name of the column in `observation_data` that corresponds to the\n        user id.\n\n    item_id : string, optional\n        The name of the column in `observation_data` that corresponds to the\n        item id.\n\n    target : string\n        The `observation_data` must contain a column of scores\n        representing ratings given by the users. If not present,\n        consider using the ranking version of the factorization model,\n        RankingFactorizationRecommender,\n        :class:`turicreate.recommender.ranking_factorization_recommender.RankingFactorizationRecommender`\n\n    user_data : SFrame, optional\n        Side information for the users.  This SFrame must have a column with\n        the same name as what is specified by the `user_id` input parameter.\n        `user_data` can provide any amount of additional user-specific\n        information.\n\n    item_data : SFrame, optional\n        Side information for the items.  This SFrame must have a column with\n        the same name as what is specified by the `item_id` input parameter.\n        `item_data` can provide any amount of additional item-specific\n        information.\n\n    num_factors : int, optional\n        Number of latent factors.\n\n    regularization : float, optional\n        Regularization for interaction terms. The type of regularization is L2.\n        Default: 1e-8; a typical range for this parameter is between 1e-12 and 1.\n\n    linear_regularization : float, optional\n        Regularization for linear term.\n        Default: 1e-10; a typical range for this parameter is between 1e-12 and 1.\n\n    side_data_factorization : boolean, optional\n        Use factorization for modeling any additional features beyond the user\n        and item columns. If True, and side features or any additional columns are\n        present, then a Factorization Machine model is trained. Otherwise, only\n        the linear terms are fit to these features.  See\n        :class:`turicreate.recommender.factorization_recommender.FactorizationRecommender`\n        for more information. Default: True.\n\n    nmf : boolean, optional\n        Use nonnegative matrix factorization, which forces the factors to be\n        nonnegative. Disables linear and intercept terms.\n\n    binary_target : boolean, optional\n        Assume the target column is composed of 0\'s and 1\'s. If True, use\n        logistic loss to fit the model.\n\n    max_iterations : int, optional\n        The training algorithm will make at most this many iterations through\n        the observed data. Default: 50.\n\n    sgd_step_size : float, optional\n        Step size for stochastic gradient descent. Smaller values generally\n        lead to more accurate models that take more time to train. The\n        default setting of 0 means that the step size is chosen by trying\n        several options on a small subset of the data.\n\n    random_seed :  int, optional\n        The random seed used to choose the initial starting point for\n        model training. Note that some randomness in the training is\n        unavoidable, so models trained with the same random seed may still\n        differ slightly. Default: 0.\n\n    solver : string, optional\n        Name of the solver to be used to solve the regression. See the\n        references for more detail on each solver. The available solvers for this\n        model are:\n\n        - *auto (default)*: automatically chooses the best solver for the data\n                              and model parameters.\n        - *sgd*:            Stochastic Gradient Descent.\n        - *adagrad*:        Adaptive Gradient Stochastic Gradient Descent [1].\n        - *als*:            Alternating Least Squares.\n\n    verbose : bool, optional\n        Enables verbose output.\n\n    kwargs : optional\n        Optional advanced keyword arguments passed in to the model\n        optimization procedure. These parameters do not typically\n        need to be changed.\n\n    Examples\n    --------\n    **Basic usage**\n\n    >>> sf = turicreate.SFrame({\'user_id\': ["0", "0", "0", "1", "1", "2", "2", "2"],\n    ...                       \'item_id\': ["a", "b", "c", "a", "b", "b", "c", "d"],\n    ...                       \'rating\': [1, 3, 2, 5, 4, 1, 4, 3]})\n    >>> m1 = turicreate.factorization_recommender.create(sf, target=\'rating\')\n\n    When a target column is present, :meth:`~turicreate.recommender.create`\n    defaults to creating a :class:`~turicreate.recommender.factorization_recommender.FactorizationRecommender`.\n\n    **Including side features**\n\n    >>> user_info = turicreate.SFrame({\'user_id\': ["0", "1", "2"],\n    ...                              \'name\': ["Alice", "Bob", "Charlie"],\n    ...                              \'numeric_feature\': [0.1, 12, 22]})\n    >>> item_info = turicreate.SFrame({\'item_id\': ["a", "b", "c", "d"],\n    ...                              \'name\': ["item1", "item2", "item3", "item4"],\n    ...                              \'dict_feature\': [{\'a\' : 23}, {\'a\' : 13},\n    ...                                               {\'b\' : 1},\n    ...                                               {\'a\' : 23, \'b\' : 32}]})\n    >>> m2 = turicreate.factorization_recommender.create(sf, target=\'rating\',\n    ...                                                user_data=user_info,\n    ...                                                item_data=item_info)\n\n    **Using the Alternating Least Squares (ALS) solver**\n\n    The factorization model can also be solved using alternating least\n    squares (ALS) as a solver option.  This solver does not support side\n    columns or other similar features.\n\n    >>> m3 = turicreate.factorization_recommender.create(sf, target=\'rating\',\n                                                                solver = \'als\')\n\n    See Also\n    --------\n    RankingFactorizationRecommender,\n    :class:`turicreate.recommender.ranking_factorization_recommender.RankingFactorizationRecommender`\n\n    References\n    -----------\n\n    [1] Duchi, John, Elad Hazan, and Yoram Singer. "Adaptive subgradient methods for online\n    learning and stochastic optimization." The Journal of Machine Learning Research 12 (2011).\n\n    '
    from turicreate._cython.cy_server import QuietProgress
    if not isinstance(observation_data, _SFrame):
        raise TypeError('observation_data input must be a SFrame')
    opts = {}
    model_proxy = _turicreate.extensions.factorization_recommender()
    model_proxy.init_options(opts)
    if user_data is None:
        user_data = _turicreate.SFrame()
    if item_data is None:
        item_data = _turicreate.SFrame()
    opts = {'user_id': user_id, 'item_id': item_id, 'target': target, 'random_seed': random_seed, 'num_factors': num_factors, 'regularization': regularization, 'linear_regularization': linear_regularization, 'binary_target': binary_target, 'max_iterations': max_iterations, 'sgd_step_size': sgd_step_size, 'solver': solver, 'side_data_factorization': side_data_factorization, 'nmf': nmf}
    if kwargs:
        try:
            possible_args = set(_get_default_options()['name'])
        except (RuntimeError, KeyError):
            possible_args = set()
        bad_arguments = set(kwargs.keys()).difference(possible_args)
        if bad_arguments:
            raise TypeError('Bad Keyword Arguments: ' + ', '.join(bad_arguments))
        opts.update(kwargs)
    extra_data = {'nearest_items': _turicreate.SFrame()}
    with QuietProgress(verbose):
        model_proxy.train(observation_data, user_data, item_data, opts, extra_data)
    return FactorizationRecommender(model_proxy)
_get_default_options = _get_default_options_wrapper('factorization_recommender', 'recommender.factorization_recommender', 'FactorizationRecommender')

class FactorizationRecommender(_Recommender):
    """
    A FactorizationRecommender learns latent factors for each
    user and item and uses them to make rating predictions.

    FactorizationRecommender [Koren_et_al]_ contains a number of options that
    tailor to a variety of datasets and evaluation metrics, making this one of
    the most powerful model in the Turi Create recommender toolkit.

    **Side information**

    Side features may be provided via the `user_data` and `item_data` options
    when the model is created.

    Additionally, observation-specific information, such as the time of day when
    the user rated the item, can also be included. Any column in the
    `observation_data` SFrame that is not the user id, item id, or target is
    treated as a observation side features. The same side feature columns must
    be present when calling :meth:`predict`.

    Side features may be numeric or categorical. User ids and item ids are
    treated as categorical variables. For the additional side features, the type
    of the :class:`~turicreate.SFrame` column determines how it's handled: strings
    are treated as categorical variables and integers and floats are treated as
    numeric variables. Dictionaries and numeric arrays are also supported.

    **Creating a FactorizationRecommender**

    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.recommender.factorization_recommender.create`
    to create an instance
    of this model. A detailed list of parameter options and code samples
    are available in the documentation for the create function.

    **Model parameters**

    Trained model parameters may be accessed using
    `m.get('coefficients')` or equivalently `m['coefficients']`.

    See Also
    --------
    create, :func:`turicreate.recommender.ranking_factorization_recommender.create`

    Notes
    -----
    **Model Definition**

    `FactorizationRecommender` trains a model capable of predicting a score for
    each possible combination of users and items.  The internal coefficients of
    the model are learned from known scores of users and items.
    Recommendations are then based on these scores.

    In the two factorization models, users and items are represented by weights
    and factors.  These model coefficients are learned during training.
    Roughly speaking, the weights, or bias terms, account for a user or item's
    bias towards higher or lower ratings.  For example, an item that is
    consistently rated highly would have a higher weight coefficient associated
    with them.  Similarly, an item that consistently receives below average
    ratings would have a lower weight coefficient to account for this bias.

    The factor terms model interactions between users and items.  For example,
    if a user tends to love romance movies and hate action movies, the factor
    terms attempt to capture that, causing the model to predict lower scores
    for action movies and higher scores for romance movies.  Learning good
    weights and factors is controlled by several options outlined below.

    More formally, when side data is not present, the predicted score for user
    :math:`i` on item :math:`j` is given by

    .. math::
       \\operatorname{score}(i, j) =
          \\mu + w_i + w_j
          + \\mathbf{a}^T \\mathbf{x}_i + \\mathbf{b}^T \\mathbf{y}_j
          + {\\mathbf u}_i^T {\\mathbf v}_j,

    where :math:`\\mu` is a global bias term, :math:`w_i` is the weight term for
    user :math:`i`, :math:`w_j` is the weight term for item :math:`j`,
    :math:`\\mathbf{x}_i` and :math:`\\mathbf{y}_j` are respectively the user and
    item side feature vectors, and :math:`\\mathbf{a}` and :math:`\\mathbf{b}`
    are respectively the weight vectors for those side features.
    The latent factors, which are vectors of length ``num_factors``, are given
    by :math:`{\\mathbf u}_i` and :math:`{\\mathbf v}_j`.

    When `binary_target=True`, the above score is passed through a logistic
    function:

    .. math::

       \\operatorname{score}(i, j) = 1 / (1 + exp (- z)),

    where :math:`z` is the original linear score.

    **Training the model**

    Formally, the objective function we are optimizing for is:

    .. math::
      \\min_{\\mathbf{w}, \\mathbf{a}, \\mathbf{b}, \\mathbf{V}, \\mathbf{U}}
      \\frac{1}{|\\mathcal{D}|} \\sum_{(i,j,r_{ij}) \\in \\mathcal{D}}
      \\mathcal{L}(\\operatorname{score}(i, j), r_{ij})
      + \\lambda_1 (\\lVert {\\mathbf w} \\rVert^2_2 + || {\\mathbf a} ||^2_2 + || {\\mathbf b} ||^2_2 )
      + \\lambda_2 \\left(\\lVert {\\mathbf U} \\rVert^2_2
                           + \\lVert {\\mathbf V} \\rVert^2_2 \\right)

    where :math:`\\mathcal{D}` is the observation dataset, :math:`r_{ij}` is the
    rating that user :math:`i` gave to item :math:`j`,
    :math:`{\\mathbf U} = ({\\mathbf u}_1, {\\mathbf u}_2, ...)` denotes the user's
    latent factors and :math:`{\\mathbf V} = ({\\mathbf v}_1, {\\mathbf v}_2, ...)`
    denotes the item latent factors.  The loss function
    :math:`\\mathcal{L}(\\hat{y}, y)` is :math:`(\\hat{y} - y)^2` by default.
    :math:`\\lambda_1` denotes the `linear_regularization` parameter and
    :math:`\\lambda_2` the `regularization` parameter.

    The model is trained using one of the following solvers:

    (a) Stochastic Gradient Descent [sgd]_ with additional tricks [Bottou]_
    to improve convergence. The optimization is done in parallel
    over multiple threads. This procedure is inherently random, so different
    calls to `create()` may return slightly different models, even with the
    same `random_seed`.

    (b) Alternating least squares (ALS), where the user latent factors are
    computed by fixing the item latent factors and vice versa.

    The Factorization Machine recommender model approximates target rating
    values as a weighted combination of user and item latent factors, biases,
    side features, and their pairwise combinations.

    The Factorization Machine [Rendle]_ is a generalization of Matrix Factorization.
    In particular, while Matrix Factorization learns latent factors for only the
    user and item interactions, the Factorization Machine learns latent factors
    for all variables, including side features, and also allows for interactions
    between all pairs of variables. Thus the Factorization Machine is capable of
    modeling complex relationships in the data. Typically,
    using `linear_side_features=True` performs better in terms of RMSE,
    but may require a longer training time.


    References
    ----------
    .. [Koren_et_al] Koren, Yehuda, Robert Bell and Chris Volinsky. `"Matrix
        Factorization Techniques for Recommender Systems." <http://www2.research
        .att.com/~volinsky/papers/ieeecomputer.pdf?utm_source=twitterfeed&utm_me
        dium=twitter>`_ Computer Volume: 42, Issue: 8 (2009): 30-37.

    .. [sgd] `Wikipedia - Stochastic gradient descent
        <http://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_

    .. [Bottou] Leon Bottou, `"Stochastic Gradient Tricks,"
        <http://research.microsoft.com/apps/pubs/default.aspx?id=192769>`_
        Neural Networks, Tricks of the Trade, Reloaded, 430--445, Lecture Notes
        in Computer Science (LNCS 7700), Springer, 2012.

    .. [Rendle] Steffen Rendle, `"Factorization Machines,"
        <http://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf>`_ in
        Proceedings of the 10th IEEE International Conference on Data Mining
        (ICDM), 2010.

    """

    def __init__(self, model_proxy):
        if False:
            print('Hello World!')
        self.__proxy__ = model_proxy

    @classmethod
    def _native_name(cls):
        if False:
            i = 10
            return i + 15
        return 'factorization_recommender'