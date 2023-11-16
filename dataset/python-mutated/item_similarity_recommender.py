"""
Methods for creating models that rank items according to their similarity
to other items.
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import turicreate as _turicreate
from turicreate.toolkits.recommender.util import _Recommender
from turicreate.toolkits._model import _get_default_options_wrapper
from turicreate.data_structures.sframe import SFrame as _SFrame

def create(observation_data, user_id='user_id', item_id='item_id', target=None, user_data=None, item_data=None, nearest_items=None, similarity_type='jaccard', threshold=0.001, only_top_k=64, verbose=True, target_memory_usage=8 * 1024 * 1024 * 1024, **kwargs):
    if False:
        print('Hello World!')
    '\n    Create a recommender that uses item-item similarities based on\n    users in common.\n\n    Parameters\n    ----------\n    observation_data : SFrame\n        The dataset to use for training the model. It must contain a column of\n        user ids and a column of item ids. Each row represents an observed\n        interaction between the user and the item.  The (user, item) pairs\n        are stored with the model so that they can later be excluded from\n        recommendations if desired. It can optionally contain a target ratings\n        column. All other columns are interpreted by the underlying model as\n        side features for the observations.\n\n        The user id and item id columns must be of type \'int\' or \'str\'. The\n        target column must be of type \'int\' or \'float\'.\n\n    user_id : string, optional\n        The name of the column in `observation_data` that corresponds to the\n        user id.\n\n    item_id : string, optional\n        The name of the column in `observation_data` that corresponds to the\n        item id.\n\n    target : string, optional\n        The `observation_data` can optionally contain a column of scores\n        representing ratings given by the users. If present, the name of this\n        column may be specified variables `target`.\n\n    user_data : SFrame, optional\n        Side information for the users.  This SFrame must have a column with\n        the same name as what is specified by the `user_id` input parameter.\n        `user_data` can provide any amount of additional user-specific\n        information. (NB: This argument is currently ignored by this model.)\n\n    item_data : SFrame, optional\n        Side information for the items.  This SFrame must have a column with\n        the same name as what is specified by the `item_id` input parameter.\n        `item_data` can provide any amount of additional item-specific\n        information. (NB: This argument is currently ignored by this model.)\n\n    similarity_type : {\'jaccard\', \'cosine\', \'pearson\'}, optional\n        Similarity metric to use. See ItemSimilarityRecommender for details.\n        Default: \'jaccard\'.\n\n    threshold : float, optional\n        Predictions ignore items below this similarity value.\n        Default: 0.001.\n\n    only_top_k : int, optional\n        Number of similar items to store for each item. Default value is\n        64.  Decreasing this  decreases the amount of memory required for the\n        model, but may also decrease the accuracy.\n\n    nearest_items : SFrame, optional\n        A set of each item\'s nearest items. When provided, this overrides\n        the similarity computed above.\n        See Notes in the documentation for ItemSimilarityRecommender.\n        Default: None.\n\n    target_memory_usage : int, optional\n        The target memory usage for the processing buffers and lookup\n        tables.  The actual memory usage may be higher or lower than this,\n        but decreasing this decreases memory usage at the expense of\n        training time, and increasing this can dramatically speed up the\n        training time.  Default is 8GB = 8589934592.\n\n    seed_item_set_size : int, optional\n        For users that have not yet rated any items, or have only\n        rated uniquely occurring items with no similar item info,\n        the model seeds the user\'s item set with the average\n        ratings of the seed_item_set_size most popular items when\n        making predictions and recommendations.  If set to 0, then\n        recommendations based on either popularity (no target present)\n        or average item score (target present) are made in this case.\n\n    nearest_neighbors_interaction_proportion_threshold : (advanced) float\n        Any item that has was rated by more than this proportion of\n        users is  treated by doing a nearest neighbors search.  For\n        frequent items, this  is almost always faster, but it is slower\n        for infrequent items.  Furthermore, decreasing this causes more\n        items to be processed using the nearest neighbor path, which may\n        decrease memory requirements.\n\n    degree_approximation_threshold : (advanced) int, optional\n        Users with more than this many item interactions may be\n        approximated.  The approximation is done by a combination of\n        sampling and choosing the interactions likely to have the most\n        impact on the model.  Increasing this can increase the training time\n        and may or may not increase the quality of the model.  Default = 4096.\n\n    max_data_passes : (advanced) int, optional\n        The maximum number of passes through the data allowed in\n        building the similarity lookup tables.  If it is not possible to\n        build the recommender in this many passes (calculated before\n        that stage of training), then additional approximations are\n        applied; namely decreasing degree_approximation_threshold.  If\n        this is not possible, an error is raised.  To decrease the\n        number of passes required, increase target_memory_usage or\n        decrease nearest_neighbors_interaction_proportion_threshold.\n        Default = 1024.\n\n    Examples\n    --------\n    Given basic user-item observation data, an\n    :class:`~turicreate.recommender.item_similarity_recommender.ItemSimilarityRecommender` is created:\n\n    >>> sf = turicreate.SFrame({\'user_id\': [\'0\', \'0\', \'0\', \'1\', \'1\', \'2\', \'2\', \'2\'],\n    ...                       \'item_id\': [\'a\', \'b\', \'c\', \'a\', \'b\', \'b\', \'c\', \'d\']})\n    >>> m = turicreate.item_similarity_recommender.create(sf)\n    >>> recs = m.recommend()\n\n    When a target is available, one can specify the desired similarity. For\n    example we may choose to use a cosine similarity, and use it to make\n    predictions or recommendations.\n\n    >>> sf2 = turicreate.SFrame({\'user_id\': [\'0\', \'0\', \'0\', \'1\', \'1\', \'2\', \'2\', \'2\'],\n    ...                        \'item_id\': [\'a\', \'b\', \'c\', \'a\', \'b\', \'b\', \'c\', \'d\'],\n    ...                        \'rating\': [1, 3, 2, 5, 4, 1, 4, 3]})\n    >>> m2 = turicreate.item_similarity_recommender.create(sf2, target="rating",\n    ...                                                  similarity_type=\'cosine\')\n    >>> m2.predict(sf)\n    >>> m2.recommend()\n\n    Notes\n    -----\n    Currently, :class:`~turicreate.recommender.item_similarity_recommender.ItemSimilarityRecommender`\n    does not leverage the use of side features `user_data` and `item_data`.\n\n    **Incorporating pre-defined similar items**\n\n    For item similarity models, one may choose to provide user-specified\n    nearest neighbors graph using the keyword argument `nearest_items`. This is\n    an SFrame containing, for each item, the nearest items and the similarity\n    score between them. If provided, these item similarity scores are used for\n    recommendations. The SFrame must contain (at least) three columns:\n\n    * \'item_id\': a column with the same name as that provided to the `item_id`\n      argument (which defaults to the string "item_id").\n    * \'similar\': a column containing the nearest items for the given item id.\n      This should have the same type as the `item_id` column.\n    * \'score\': a numeric score measuring how similar these two items are.\n\n    For example, suppose you first create an ItemSimilarityRecommender and use\n    :class:`~turicreate.recommender.ItemSimilarityRecommender.get_similar_items`:\n\n    >>> sf = turicreate.SFrame({\'user_id\': ["0", "0", "0", "1", "1", "2", "2", "2"],\n    ...                       \'item_id\': ["a", "b", "c", "a", "b", "b", "c", "d"]})\n    >>> m = turicreate.item_similarity_recommender.create(sf)\n    >>> nn = m.get_similar_items()\n    >>> m2 = turicreate.item_similarity_recommender.create(sf, nearest_items=nn)\n\n    With the above code, the item similarities computed for model `m` can be\n    used to create a new recommender object, `m2`. Note that we could have\n    created `nn` from some other means, but now use `m2` to make\n    recommendations via `m2.recommend()`.\n\n\n    See Also\n    --------\n    ItemSimilarityRecommender\n\n    '
    from turicreate._cython.cy_server import QuietProgress
    if not isinstance(observation_data, _SFrame):
        raise TypeError('observation_data input must be a SFrame')
    opts = {}
    model_proxy = _turicreate.extensions.item_similarity()
    model_proxy.init_options(opts)
    if user_data is None:
        user_data = _turicreate.SFrame()
    if item_data is None:
        item_data = _turicreate.SFrame()
    if nearest_items is None:
        nearest_items = _turicreate.SFrame()
    opts = {'user_id': user_id, 'item_id': item_id, 'target': target, 'similarity_type': similarity_type, 'threshold': threshold, 'target_memory_usage': float(target_memory_usage), 'max_item_neighborhood_size': only_top_k}
    extra_data = {'nearest_items': nearest_items}
    if kwargs:
        try:
            possible_args = set(_get_default_options()['name'])
        except (RuntimeError, KeyError):
            possible_args = set()
        bad_arguments = set(kwargs.keys()).difference(possible_args)
        if bad_arguments:
            raise TypeError('Bad Keyword Arguments: ' + ', '.join(bad_arguments))
        opts.update(kwargs)
    extra_data = {'nearest_items': nearest_items}
    opts.update(kwargs)
    with QuietProgress(verbose):
        model_proxy.train(observation_data, user_data, item_data, opts, extra_data)
    return ItemSimilarityRecommender(model_proxy)
_get_default_options = _get_default_options_wrapper('item_similarity', 'recommender.item_similarity', 'ItemSimilarityRecommender')

class ItemSimilarityRecommender(_Recommender):
    """
    A model that ranks an item according to its similarity to other items
    observed for the user in question.

    **Creating an ItemSimilarityRecommender**

    This model cannot be constructed directly.  Instead, use
    :func:`turicreate.recommender.item_similarity_recommender.create`
    to create an instance
    of this model. A detailed list of parameter options and code samples
    are available in the documentation for the create function.

    Notes
    -----
    **Model Definition**

    This model first computes the similarity
    between items using the observations of users who have interacted with both
    items. Given a similarity between item :math:`i` and :math:`j`,
    :math:`S(i,j)`, it scores an item :math:`j` for user :math:`u` using a
    weighted average of the user's previous observations :math:`I_u`.

    There are three choices of similarity metrics to use: 'jaccard',
    'cosine' and 'pearson'.

    `Jaccard similarity
    <http://en.wikipedia.org/wiki/Jaccard_index>`_
    is used to measure the similarity between two set of elements.
    In the context of recommendation, the Jaccard similarity between two
    items is computed as

    .. math:: \\mbox{JS}(i,j)
            = \\frac{|U_i \\cap U_j|}{|U_i \\cup U_j|}

    where :math:`U_{i}` is the set of users who rated item :math:`i`.
    Jaccard is a good choice when one only has implicit feedbacks of items
    (e.g., people rated them or not), or when one does not care about how
    many stars items received.

    If one needs to compare the ratings of items, Cosine and Pearson similarity
    are recommended.

    The Cosine similarity between two items is computed as

    .. math:: \\mbox{CS}(i,j)
            = \\frac{\\sum_{u\\in U_{ij}} r_{ui}r_{uj}}
                {\\sqrt{\\sum_{u\\in U_{i}} r_{ui}^2}
                 \\sqrt{\\sum_{u\\in U_{j}} r_{uj}^2}}

    where :math:`U_{i}` is the set of users who rated item :math:`i`,
    and :math:`U_{ij}` is the set of users who rated both items :math:`i` and
    :math:`j`. A problem with Cosine similarity is that it does not consider
    the differences in the mean and variance of the ratings made to
    items :math:`i` and :math:`j`.

    Another popular measure that compares ratings where the effects of means and
    variance have been removed is Pearson Correlation similarity:

    .. math:: \\mbox{PS}(i,j)
            = \\frac{\\sum_{u\\in U_{ij}} (r_{ui} - \\bar{r}_i)
                                        (r_{uj} - \\bar{r}_j)}
                {\\sqrt{\\sum_{u\\in U_{ij}} (r_{ui} - \\bar{r}_i)^2}
                 \\sqrt{\\sum_{u\\in U_{ij}} (r_{uj} - \\bar{r}_j)^2}}

    The predictions of items depend on whether `target` is specified.
    When the `target` is absent, a prediction for item :math:`j` is made via

    .. math:: y_{uj}
            = \\frac{\\sum_{i \\in I_u} \\mbox{SIM}(i,j)  }{|I_u|}


    Otherwise, predictions for ``jaccard`` and ``cosine`` similarities are made via

    .. math:: y_{uj}
            = \\frac{\\sum_{i \\in I_u} \\mbox{SIM}(i,j) r_{ui} }{\\sum_{i \\in I_u} \\mbox{SIM}(i,j)}

    Predictions for ``pearson`` similarity are made via

    .. math:: y_{uj}
            = \\bar{r}_j + \\frac{\\sum_{i \\in I_u} \\mbox{SIM}(i,j) (r_{ui} - \\bar{r}_i) }{\\sum_{i \\in I_u} \\mbox{SIM}(i,j)}


    For more details of item similarity methods, please see, e.g.,
    Chapter 4 of [Ricci_et_al]_.

    See Also
    --------
    create

    References
    ----------
    .. [Ricci_et_al] Francesco Ricci, Lior Rokach, and Bracha Shapira.
        `Introduction to recommender systems handbook
        <http://www.inf.unibz.it/~ricci/papers
        /intro-rec-sys-handbook.pdf>`_. Springer US, 2011.
    """

    def __init__(self, model_proxy):
        if False:
            for i in range(10):
                print('nop')
        '__init__(self)'
        self.__proxy__ = model_proxy

    @classmethod
    def _native_name(cls):
        if False:
            i = 10
            return i + 15
        return 'item_similarity'